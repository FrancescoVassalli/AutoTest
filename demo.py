import openai
import os
import json
openai.api_key = os.getenv("OPEN_API_KEY")
base_prompt='''Play as a software quality assurance expert. 
Given software requirements respond with a written verification plan and use function calls to support that plan.
The user will continue to respond with the result of the function calls until the plan is executed. 
Once the plan is executed review the results and determine which requirements were met.'''
def addOne(input):
    return input+1

def addOneHelper(arguments):
    front = arguments.find(': ')
    arguments_drop_front = arguments[front+2:]
    parameter = arguments_drop_front[:arguments_drop_front.find('\n')]
    return addOne(int(parameter))

function_dict = {'addOne':addOneHelper}

def grade(requirements):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{'role': 'system', "content": base_prompt}, {'role': 'user', 'content': requirements},],
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
def getFunctionCall(completion):
    function_call = completion.choices[0].message.function_call
    name = function_call.name
    arguments = function_call.arguments
    return (name,arguments)

def getAssistantInfo(completion):
    return completion.choices[0].message.content

def converse(function_call,function_result, assistantInfo,requirements):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{'role': 'system', "content": base_prompt},
                  {'role': 'user', 'content': requirements},
                  {'role': 'assistant', 'content':assistantInfo,'function_call':function_call},
                  {'role': 'user', 'content':'Function Result: '+str(function_result)}],
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

requirement = "Requirement: The library can add one to inputs from the user."
completion = grade(requirement)
print(completion)
original_function_call = completion.choices[0].message.function_call
function_call = getFunctionCall(completion)
assistantInfo = getAssistantInfo(completion)
function_result = function_dict[function_call[0]](function_call[1])
print(converse(original_function_call,function_result,assistantInfo,requirement))