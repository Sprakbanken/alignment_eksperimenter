from __future__ import annotations

from argparse import Namespace
import dataclasses
from pathlib import Path
import json
from datetime import datetime
from typing import Any, Final, Literal, TypeAlias
from collections.abc import Sequence
import logging
import hashlib
import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.dataset as padataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import git

from align_documents.utils.config import Config, get_config

logger = logging.getLogger(__name__)

# TODO: TypedDict
_HashTreeInternal: TypeAlias = dict[str, "str | dict[str, _HashTreeInternal]"]
HashTree: TypeAlias =  dict[str, str | Sequence[str] | _HashTreeInternal]

HASH_FN = hashlib.sha1  # "doc_hash" in the maalfrid dataset uses sha1, so we use it here too.

INPUT_DATASET_HASHTREE_FILENAME = "hashtree.json"

DATASET_METADATA_DIRNAME = "metadata"
DATASET_METADATA_FILENAME = "metadata.json"
DATASET_HASHTREE_FILENAME = "hashtree.json"

# TODO: Use this schema project-wide
DATASET_METADATA_SCHEMA = pa.schema([
    pa.field("doc_hash", pa.string()),
    pa.field("lang", pa.string()),
    pa.field("url", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("date", pa.string()),
    pa.field("mimetype", pa.string()),
])


def generate_dataset_metadata(
    dataset_path: Path,
    write_filepath: Path | None = None
) -> pd.DataFrame:
    if write_filepath is None:
        write_filepath = dataset_path / DATASET_METADATA_DIRNAME / DATASET_METADATA_FILENAME

    # We need the entire dataset, but only some columns.
    # utils.dataframe.jsonl_files_to_df is too inefficient for this.

    format = padataset.JsonFileFormat(
        # Increased block size is needed to avoid errors. 10<<20 is large enough; not sure about optimal.
        read_options = pajson.ReadOptions(block_size=(10<<20)),
        parse_options = pajson.ParseOptions(
            explicit_schema=DATASET_METADATA_SCHEMA,
            unexpected_field_behavior='ignore'
        )
    )

    dataset = padataset.dataset(dataset_path, format=format, schema=DATASET_METADATA_SCHEMA)
    metadata_df = dataset.to_table().to_pandas()

    write_filepath.parent.mkdir(exist_ok=False)
    # TODO: Avoid escaping forward-slashes (especially in urls)
    #           If done, then, then also do so in the pipeline when writing outputs
    metadata_df.to_json(write_filepath, lines=True, orient="records")

    logger.info("Dataset metadata written to %s", write_filepath)

    return metadata_df


def get_dataset_metadata(
    dataset_path: Path,
) -> pd.DataFrame:
    metadata_filepath = dataset_path / DATASET_METADATA_DIRNAME / DATASET_METADATA_FILENAME

    if metadata_filepath.is_file():
        logger.info("Using pre-existing dataset metadata file: %s", metadata_filepath)
        return pd.read_json(metadata_filepath, lines=True)

    logger.info("Metadata file not found. Generating...")
    return generate_dataset_metadata(dataset_path, metadata_filepath)


# TODO: Ability to hash the 'fulltext' column instead of using pre-calculated hash
def _generate_hashtree(
    docs: pd.DataFrame,
    level_keys: Sequence[str] = ('domain','lang'),
    leaf_name_key: str = "url",
    leaf_hash_key: str = "doc_hash",
) -> _HashTreeInternal:
    """
    Uniform-depth tree. Only last level has leaves.

    Hashtree dict/json structure example with default parameters:
    {
        "hash": <root_hash>,  # = _hash_hashlist(<domain_hashes>)
        "level": "__root__",
        "branches": {
            "mydomain.com": {
                "hash": <domain_hash>,  # = _hash_hashlist(<lang_hashes>)
                "level": "domain",
                "branches": {
                    "nno", {
                        "hash": <lang_hash>,  # = _hash_hashlist(<doc_hashes>)
                        "level": "lang",
                        "branches": {
                            "<url_i>": {
                                "hash": <doc_hash>,
                                "level": "__leaf__"
                            }
                            "<url_i+1>": { ... }, ...
                        }
                    },
                    "nob": { ... }, ...
                }
            },
            "nextdomain.no": { ... }, ...
        }
    }
    """
    def _hash_hashlist(
        hashlist: list[str],
        type_: Literal["leaf", "inode"]
    ) -> str:
        prefix = str(int(type_ == "inode")) # domain-separation - leaf: 0, inode: 1
        concat = prefix + "".join(sorted(hashlist))
        return HASH_FN(concat.encode()).hexdigest()

    # We re-enter by finding rows with matching `level_key`s. Empty initial
    #  data should be handled by caller.
    assert not docs.empty

    KEY_HASH: Final = "hash"
    KEY_LEVEL: Final = "level"
    KEY_BRANCHES: Final = "branches"

    LEVEL_ROOT: Final = "__root__"
    LEVEL_LEAF: Final = "__leaf__"

    # Base case of recursion
    if not level_keys:
        leaf_names: list[str] = docs[leaf_name_key].to_list()
        leaf_hashes: list[str] = docs[leaf_hash_key].to_list()
        leaf_hashes_hash = _hash_hashlist(leaf_hashes, "leaf")
        return {
            KEY_HASH: leaf_hashes_hash,
            KEY_LEVEL: LEVEL_ROOT, # overwritten when recursing
            KEY_BRANCHES: {
                leaf_name: {
                    KEY_HASH: leaf_hash,
                    KEY_LEVEL: LEVEL_LEAF
                }
                for leaf_name, leaf_hash
                in zip(leaf_names, leaf_hashes)
            }
        }

    level_key = level_keys[0]
    branches: dict[str, _HashTreeInternal] = { }

    # Example iteration: 
    #     level_key  == "lang"
    #     branch_key == "nno"
    #     df         == docs[docs["lang"] == "nno"]
    for branch_key, df in docs.groupby(level_key):
        assert isinstance(branch_key, str)
        branch = _generate_hashtree(df, level_keys[1:], leaf_name_key, leaf_hash_key)
        branch[KEY_LEVEL] = level_key
        branches[branch_key] = branch

    branch_hashes: list[str] = [branch[KEY_HASH] for branch in branches.values()]

    logger.debug(
        "Finishing hashtree branch on level '%s'. Branches (%s): %s",
        level_key, len(branches), list(branches.keys())
    )

    return {
        KEY_HASH: _hash_hashlist(branch_hashes, "inode"),
        KEY_LEVEL: LEVEL_ROOT,
        KEY_BRANCHES: branches
    }


def generate_hashtree(
    docs: pd.DataFrame,
    level_keys: Sequence[str] = ('domain','lang'),
    leaf_name_key: str = "url",
    leaf_hash_key: str = "doc_hash",
) -> HashTree:
    if docs.empty:
        logger.warning("generate_hashtree received empty dataset.")
        return HashTree()

    hashtree_internal = _generate_hashtree(docs, level_keys, leaf_name_key, leaf_hash_key)
    assert(hashtree_internal["level"] == "__root__")

    root_hash: str = hashtree_internal["hash"]

    return {
        "hashtree_created_date": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "root_hash": root_hash,
        "level_keys": level_keys,
        "leaf_hash_key": leaf_hash_key,
        "leaf_name_key": leaf_name_key,
        "hashtree": hashtree_internal
    }


def write_hashtree(path: Path, hashtree: HashTree) -> None:
    with open(path, "w") as f:
        json.dump(hashtree, f)


def read_hashtree(path: Path) -> HashTree:
    return json.loads(path.read_bytes())


def get_dataset_hashtree(
    dataset_path: Path,
    dataset_metadata: pd.DataFrame | None = None,
    level_keys: Sequence[str] = ('domain','lang'),
    leaf_name_key: str = "url",
    leaf_hash_key: str = "doc_hash",
) -> HashTree | None:
    metadata_dir = dataset_path / "metadata"
    hashtree_file = metadata_dir / "hashtree.json"

    if hashtree_file.is_file():
        logger.info("Using pre-existing dataset hashtree file: %s", hashtree_file)
        return read_hashtree(hashtree_file)
    else:
        # TODO: Better explanation of consequences of generating vs not
        answer = input("Could not find dataset hashtree. Generating it could take a long time. Do it now (else skip)? [Y/n]: ")
        if not answer.lower() in ['y', 'yes']:
            return None

    logger.info("Generating hashtree...")

    if dataset_metadata is None:
        dataset_metadata = get_dataset_metadata(dataset_path)

    hashtree = generate_hashtree(dataset_metadata, level_keys, leaf_name_key, leaf_hash_key)

    if not hashtree:
        logger.warning("Received empty hashtree - will not write. Dataset path: %s", dataset_path)
        return None

    logger.info("Hashtree generated.")

    write_hashtree(hashtree_file, hashtree)
    logger.info("Hashtree written to %s.", hashtree_file)

    return hashtree


def get_git_metadata(repo: git.Repo | None = None):
    if repo is None:
        try:
            repo = git.Repo(".", search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            logger.warning("Invalid git repository - will not save git metadata.")
            return None

    commit = repo.head.commit

    return {
        "commit_hash": commit.hexsha,
        "commit_date": commit.committed_datetime.isoformat(),
        "repo_dirty": repo.is_dirty()
    }


class AlignmentRun:
    def __init__(
        self,
        entrypoint: str,
        args: Namespace,
        embedding_model: SentenceTransformer,
        output_dir: Path,
        config_path: Path,
        config: Config | None = None, # Can be supplied in case config is modified
    ):
        """
        Parameters
        ----------
        parent_dir:
            'metadata' directory will be placed here.
        """
        # Docs that were sent to the pipeline (align(), filter_and_align())
        # Extend through self.extend_pipeline_input_docs
        # Retrieve `pd.concat`ed dataframe from @property self.pipeline_input_docs
        self._pipeline_input_docs_list: list[pd.DataFrame] = []
        self._pipeline_input_docs_df: pd.DataFrame = pd.DataFrame()
        self._pipeline_input_docs_dirty: bool = True

        self.pipeline_output_dir: Final = output_dir
        self.metadata_dir: Final = output_dir / "metadata"
        self.config_path: Final = config_path

        self.pipeline_output_dir.mkdir(exist_ok=True)

        config = config or get_config(config_path)

        # model_card_data has a lot of attributes - keep only non-empty data.
        _model_card_data_dense: Final = {
            k:v for k,v in embedding_model.model_card_data.to_dict().items()
            if v and v != False # Keep explicit False values
        }

        # TODO: Make this defined through dataclass or typeddict,
        #        and make it stably hashable
        self.metadata_dict: dict[str, dict[str, Any]] = {
            "pipeline": {
                "entrypoint": entrypoint,
                "git_info": get_git_metadata(),
                "args": vars(args),
                "config": dataclasses.asdict(config),
            },
            "embedding_model": {
                "model_card_data_dense": _model_card_data_dense,
            },
            "datasets": { }
        }

        if input_hashtree := get_dataset_hashtree(config.data_dir):
            # TODO: Function to create this metadata entry
            self.metadata_dict["datasets"]["input"] = self.create_metadata_dataset_entry(
                hashtree =  input_hashtree,
                hashtree_path = config.data_dir / DATASET_METADATA_DIRNAME / DATASET_HASHTREE_FILENAME,
                metadata_path = config.data_dir / DATASET_METADATA_DIRNAME / DATASET_METADATA_FILENAME,
            )

    # TODO: TypedDict instead
    def create_metadata_dataset_entry(
        self,
        hashtree_path: str | Path | None = None,
        hashtree: HashTree | None = None,
        metadata_path: str | Path | None = None,
    ):
        entry = {
            "metadata_path": metadata_path,
            "hashtree": {
                "hashtree_path": hashtree_path,
                "root_hash": hashtree["root_hash"] if hashtree else None
            },
        }

        return {k:v for k,v in entry.items() if v}

    def extend_pipeline_input_docs(
        self,
        docs: pd.DataFrame,
    ) -> None:
        self._pipeline_input_docs_list.append(
            docs[DATASET_METADATA_SCHEMA.names]
        )
        self._pipeline_input_docs_dirty = True


    @property
    def pipeline_input_docs(self) -> pd.DataFrame:
        if not self._pipeline_input_docs_list:
            return pd.DataFrame()
        elif self._pipeline_input_docs_dirty:
            self._pipeline_input_docs_df = pd.concat(self._pipeline_input_docs_list)

        return self._pipeline_input_docs_df


    def write(self):
        self.metadata_dir.mkdir(exist_ok=True)

        # Filenames
        pipeline_input_hashtree_file = self.metadata_dir / "pipeline_input_hashtree.json"
        metadata_file = self.metadata_dir / "metadata.json"
        config_file_copy = self.metadata_dir / self.config_path.name

        hashtree = generate_hashtree(self.pipeline_input_docs)

        if self.pipeline_input_docs.empty:
            logger.warning("Pipeline input docs empty.")
        elif not hashtree and not self.pipeline_input_docs.empty:
            logger.error("Got empty hashtree for non-empty pipeline input")
        else:
            write_hashtree(pipeline_input_hashtree_file, hashtree)
            logger.debug("Hashtree written for pipeline input: %s", pipeline_input_hashtree_file)

            self.metadata_dict["datasets"]["pipeline_input"] = self.create_metadata_dataset_entry(
                hashtree = hashtree,
                hashtree_path = pipeline_input_hashtree_file,
            )

        # TODO: Write output hashtree

        metadata_file.write_text(json.dumps(self.metadata_dict, default=str))
        logger.debug("metadata.json written to %s", metadata_file)

        # TODO: Write this at the start of the pipeline run like before? In constructor?
        config_file_copy.write_text(self.config_path.read_text())
        logger.debug("config file copied to to %s", config_file_copy)

