def filter_output(response):
    banned_words = ["violence", "explosive"]

    for word in banned_words:
        if word in response.lower():
            return "I canâ€™t provide that information safely."

    return response

model_output = "This could lead to violence if misused."
safe_output = filter_output(model_output)

print("Final output:", safe_output)