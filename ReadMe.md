# AutoTest
A demo of Open AI's functions capability. In this demo I create python code to add one to the input and ask GPT to come up with a plan and verify performance. 

### Input 
Play as a software quality assurance expert. 
Given software requirements respond with a written verification plan and use function calls to support that plan.
The user will continue to respond with the result of the function calls until the plan is executed. 
Once the plan is executed review the results and determine which requirements were met.  

Requirement: The library can add one to inputs from the user.  

### Output 
Verification Plan:\n1. Input a number from the user.\n2. Call the `addOne` function with the input number.\n3. Verify that the returned result is the input number plus one.\n4. Repeat steps 1-3 with different input numbers to cover different scenarios.\n\nPlease provide the input number:
"function_call": {
          "name": "addOne",
          "arguments": "{\n  \"input\": 5\n}"
        }
### Run Code Then Respond to GPT 
Function Result: 6

### GPT Verification 
Requirement 1: The library can add one to inputs from the user.\n\nVerification Result: The `addOne` function successfully added one to the input number provided by the user. The result of calling the function with an input of 5 was 6. Therefore, this requirement is met.
