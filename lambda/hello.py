def main():
    print("Running hello world app!")


def handler(event, context):
    try:
        main()
        return {"statusCode": 200, "body": "Email sent successfully!"}
    except Exception as e:
        return {
            "statusCode": 400,
            "body": f"Error: {e}",
        }
