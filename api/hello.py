"""The hello script."""
import logging
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

names = ["mole", "moje", "roley", "mojamon"]


def main() -> None:
    name = random.choice(names)
    print(f"hello {name}!")


def handler(event, context) -> dict:
    logger.info("Running handler")
    try:
        main()
        return {"statusCode": 200, "body": "Email sent successfully!"}
    except Exception as e:
        logger.exception("main() raised exception:")
        return {
            "statusCode": 400,
            "body": f"Error: {e}",
        }
