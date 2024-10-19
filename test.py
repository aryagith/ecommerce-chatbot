import pytest
from chatbot import get_chatbot_response  # Adjust the import based on your file structure

# Define the expected responses for specific inputs
test_cases = [
    ("Hi", ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help you today?"]),
    ("How are you?", ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help you today?"]),
    ("Goodbye", ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]),
    ("Thanks", ["Happy to help!", "Any time!", "My pleasure"]),
    ("What hours are you open?", ["We're open every day 9am-9pm", "Our hours are 9am-9pm every day"]),
    ("See you later", ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]),
    ("Thank you", ["Happy to help!", "Any time!", "My pleasure"]),
    ("What are your hours?", ["We're open every day 9am-9pm", "Our hours are 9am-9pm every day"]),
    ("Is anyone there?", ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help you today?"]),
    ("Good day", ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help you today?"]),
    ("That's helpful", ["Happy to help!", "Any time!", "My pleasure"]),
]

@pytest.mark.parametrize("user_input, expected_responses", test_cases)
def test_chatbot_responses(user_input, expected_responses):
    response = get_chatbot_response(user_input)
    assert response in expected_responses