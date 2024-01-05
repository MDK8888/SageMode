import os
import time
import json
from dotenv import load_dotenv
from ..ResourceUser.ResourceUser import ResourceUser
from Types.Arn import *

class DeploymentStateMachine(ResourceUser):

    def __init__(self, step_function_arn:StepFunctionArn = None):
        load_dotenv()
        role_arn = RoleArn(os.environ["STEP_FUNCTION_ROLE_ARN"])
        super().__init__(role_arn)
        self.step_function_arn = step_function_arn
        self.step_function_client = self.boto3_session.client("stepfunctions", region_name=os.environ["AWS_REGION"])
        self.state_machine_definition = {}

    def deploy(self, state_machine_name: str, comment:str=None, resource_users:list[ResourceUser]=[], deployment_args:list[dict]=[]) -> None:

        def verify_transition(index:int) -> None:
            if index >= len(resource_users)-1:
                return
            resource_user, next_resource_user = resource_users[index], resource_users[index+1]
            if not resource_user.next == next_resource_user.previous:
                raise ValueError(f"Type mismatch between the output of resource user {index} and the input of resource user {index+1}")

        def create_current_state(index:int) -> dict:
            current_dict = {"Type":"Task"}
            resource_user = resource_users[index]
            current_lambda_arn = resource_user.deploy(**(deployment_args[i]))
            current_dict["Resource"] = current_lambda_arn.raw_str
            if index != len(resource_users) - 1:
                current_dict["Next"] = f"lambda{index+1}"
            else:
                current_dict["End"] = True
            return current_dict
        
        if self.step_function_arn:
            raise ValueError("This object cannot call deploy() if it aiready has a step function arn. Set 'self.step_function_arn = None' and try again.")

        if comment:
            self.state_machine_definition["comment"] = comment
        
        self.state_machine_definition["StartAt"] = "lambda0"
        self.state_machine_definition["States"] = {}
        for i in range(len(resource_users)):
            verify_transition(i)
            current_dict = create_current_state(i)
            self.state_machine_definition["States"][f"lambda{i}"] = current_dict

        response = self.step_function_client.create_state_machine(
            name=state_machine_name,
            definition=json.dumps(self.state_machine_definition),
            roleArn=self.role_arn.raw_str
        )
        self.state_machine_arn = response["stateMachineArn"]
        print("deployment state machine created. State machine arn: ", self.state_machine_arn)
    
    def use(self, data:dict, check_timeout:int) -> dict:

        def wait_for_run_to_complete(execution_arn:str):
            execution_status = "RUNNING"
            while execution_status == 'RUNNING':
                response = self.step_function_client.describe_execution(
                    executionArn=execution_arn
                )
                execution_status = response['status']

                if execution_status == 'RUNNING':
                    # Optionally wait for a few seconds before checking again
                    time.sleep(check_timeout)
            
            return execution_status

        if not hasattr(self, "state_machine_arn"):
            raise ValueError("You did not deploy your state machine.")
        
        self.check_input(data)
        response = self.step_function_client.start_execution(
            stateMachineArn=self.state_machine_arn,
            input=json.dumps(data)
        )
        execution_arn = response['executionArn']
        execution_status = wait_for_run_to_complete(execution_arn)

        if execution_status == 'SUCCEEDED':
            output = self.step_function_client.get_execution_history(
                executionArn=execution_arn,
                reverseOrder=True,
                maxResults=1,
                includeExecutionData=True
            )['events'][0]['executionSucceededEventDetails']['output']            
            output = json.loads(output)
            self.check_output(output)
            return output
        else:
            print("Execution did not succeed. Status:", execution_status)