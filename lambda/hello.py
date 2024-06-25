"""The hello script"""
import random

names = ["mole", "moje", "roley", "mojamon"]


def main() -> None:
    name = random.choice(names)
    print(f"Hello {name}!")


def handler(event, context) -> dict:
    try:
        main()
        return {"statusCode": 200, "body": "Email sent successfully!"}
    except Exception as e:
        return {
            "statusCode": 400,
            "body": f"Error: {e}",
        }
