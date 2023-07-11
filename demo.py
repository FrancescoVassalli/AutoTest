import openai
import os
import json
openai.api_key = os.getenv("OPEN_API_KEY")
base_prompt='''Play as a software quality assurance expert. 
Given software requirements and functions to verify call the functions to verify that they meet given requirements.
The user will continue to respond with the result of the function calls until all requirements have been tested. 
Once the requirements have been tested review the results and determine which requirements were met.'''
def addOne(input):
    return input+1
def addTwo(input):
    return input+3
def addOneHelper(arguments):
    front = arguments.find(': ')
    arguments_drop_front = arguments[front+2:]
    parameter = arguments_drop_front[:arguments_drop_front.find('\n')]
    return addOne(int(parameter))

function_dict = {'addOne':addOneHelper}

def addMessage(messages, role, content, function_call):
    messageDict = {'role': role, "content": content}
    if not function_call is None:
        messageDict['function_call'] = function_call
    messages.append(messageDict)
    return messages
def grade(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[
            {
                "name": "addOne",
                "description": "Add one to the input and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "integer",
                            "description": "the input",
                        },
                    },
                    "required": ["input"],
                },
            },
            {
                "name": "addTwo",
                "description": "Add two to the input and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "integer",
                            "description": "the input",
                        },
                    },
                    "required": ["input"],
                },
            }
        ],
        function_call="auto",
        temperature=0
    )
    return completion
def getFunctionCall(completion):
    function_call = completion.choices[0].message.function_call
    name = function_call.name
    arguments = function_call.arguments
    return (name,arguments)

def getAssistantInfo(completion):
    return completion.choices[0].message.content

def converse(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[
            {
                "name": "addOne",
                "description": "Add one to the input and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "integer",
                            "description": "the input",
                        },
                    },
                    "required": ["input"],
                },
            }
        ],
        function_call="auto",
        temperature=0
    )
    return completion

requirement = "Requirement: The library can add one or two to inputs from the user."
messages = [{'role': 'system', "content": base_prompt}, {'role': 'user', 'content': requirement},]
completion = grade(messages)
print(completion)
original_function_call = completion.choices[0].message.function_call
assistantInfo = getAssistantInfo(completion)
messages =addMessage(messages,'assistant',assistantInfo,original_function_call)
function_call = getFunctionCall(completion)
function_result = function_dict[function_call[0]](function_call[1])
messages = addMessage(messages,'user','Function Result: '+str(function_result),None)
i=1
while i<10:
    completion = converse(messages)
    print('\n\n\n'+str(i)+'\n\n\n')
    print(completion)
    original_function_call = completion.choices[0].message.function_call
    assistantInfo = getAssistantInfo(completion)
    messages = addMessage(messages, 'assistant', assistantInfo, original_function_call)
    function_call = getFunctionCall(completion)
    function_result = function_dict[function_call[0]](function_call[1])
    messages = addMessage(messages, 'user', 'Function Result: ' + str(function_result),None)
    i=i+1