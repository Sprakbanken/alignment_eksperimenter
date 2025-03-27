from main.parser import get_parser
from main.logging import setup_logging
from main.config import get_config

def main():
    parser = get_parser()
    args = parser.parse_args()

    setup_logging(args.command, args.log_level)
    config = get_config(args.config_file)

    args.func(args, config)

if __name__ == "__main__":
    main()
