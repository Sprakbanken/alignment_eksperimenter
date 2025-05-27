import json
import argparse
import os

def get_unique_domain_pairs(input_path, output_dir, output_file, limit=20):
    seen_domains = set()
    unique_pairs = []

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r') as infile:
        for line in infile:

            pair = json.loads(line)
            domain_nno = pair.get("domain_nno")
            domain_nob = pair.get("domain_nob")

            domain_key = frozenset([domain_nno, domain_nob])
            if domain_nno and domain_nob and domain_key not in seen_domains:
                unique_pairs.append(pair)
                seen_domains.add(domain_key)

            if len(unique_pairs) == limit:
                break

    with open(f"{output_dir+'/'+output_file}", 'w') as outfile:
        for pair in unique_pairs:
            json.dump(pair, outfile)
            outfile.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output_dir', '-o')
    parser.add_argument('--output_file', '-of')
    parser.add_argument('--limit', '-l', type=int, default=20)

    args = parser.parse_args()

    get_unique_domain_pairs(args.input, args.output_dir, args.output_file, args.limit)

if __name__ == "__main__":
    main()
