# SageMode
SageMode is a python library for deploying, scaling, and monitoring machine learning models and LLMs in particular at scale. It is native to AWS, which means that SageMode uses boto3 under the hood to interact with services like EC2, S3, SageMaker, and Lambda.       

If you like this project, give it a star ⭐! It helps out a lot 😁

# Problems SageMode solves
✅ Standardized but also flexible deployments of both Huggingface and PyTorch models on either SageMaker or EC2.   
✅ Custom pipelines for processing both pre and post inference are supported.   
✅ You can deploy LLM models to AWS in as few as 5 lines of code!  
✅ Wraps all inference endpoints around Lambda, meaning that scalability and low cost are built in.  
✅ Supports the chaining of PyTorch or Huggingface Models in a similar manner to Langchain, native to AWS.  
✅ (Not yet supported) High LLM inference speeds with quantization, GPT-Fast, and vllm.   
✅ (Not yet supported) Scale your LLM deployments up and down with high speed and low cost.  
✅ (Not yet supported) High observability into your LLMs in production with Datadog/Grafana and WhyLabs.  

# Quickstart
- Make sure that you have at least Python version 3.10.2 on your machine.
- Make a virtual environment with the python command `python -m venv <<venv name>>`.
- If you are on Windows, run the command `./<<venv name>>/Scripts/activate`. If you are on a Mac, run `source <<venv name>>/bin/activate`.
- Let's get the party started 🎉! Run `pip install sagemode`.
  
- Role Configuration: The following will assume that you have no previous experience with AWS. In addition, many steps have links to images which are visual aids.
  1. Create an AWS Account: Go to https://aws.amazon.com/console/ and create a root account and follow the steps as necessary, adding MFA and entering your card info.
  2. Log in to your AWS root account and enter 'IAM' into the search bar like [so](https://github.com/MDK8888/SageMode/assets/79173446/36c830a9-d2d2-44a9-a617-d145d36cef5a):
  3. Click on the first result, and then you should see a tab with roles and users. Click on [users](https://github.com/MDK8888/SageMode/assets/79173446/54d6f8c3-af7d-4187-856f-831b77b79f4f)
  4. Click on the orange 'Create user' button in the top right corner, write a name for your user, and then click on orange 'Next' button in the bottom right.
  5. You should now see options for [role configuration](https://github.com/MDK8888/SageMode/assets/79173446/931bb503-b341-4414-bd1d-cd229ff7f212). Make sure you select 'Attach policies directly.'
  6. Ignore everything and scroll down to click on the orange 'Next' button.
  7. Ignore everything and click on the orange 'Create user' [button](https://github.com/MDK8888/SageMode/assets/79173446/e7dc961c-96e8-4a3c-b7ef-624be6a20aff) in the bottom right.
  8. Congratulations, you have just created your first IAM user 🎊! You should now be back on the 'Users' page. Click on the role you just created, and its ARN should [appear](https://github.com/MDK8888/SageMode/assets/79173446/3680c0c8-4132-488e-aa07-2bf6561fc1d3).
  9. Copy the above arn somewhere. Then, click on the 'Add permissions' button in gray. Click on the 'Create inline policy' button that appears.
  10. You should now see a tab where you can switch between 'Visual' and 'JSON'. Select 'JSON' and copy in the following JSON, where your role arn from the previous step appears as a string.
      ```
      {
      	"Version": "2012-10-17",
      	"Statement": [
      		{
      			"Sid": "VisualEditor0",
      			"Effect": "Allow",
      			"Action": [
      				"iam:GetRole",
      				"iam:PassRole"
      			],
      			"Resource": [
      				"arn:aws:iam::..."
      			]
      		}
      	]
      }
      ```
  11. Scroll down and click the orange 'Next' button.
  12. Enter a meaningful name for your policy, and then click on the orange 'Create policy' [button](https://github.com/MDK8888/SageMode/assets/79173446/eb309043-c255-44fb-bc86-d69e8a808e92).
  13. You should now be back on this [page](https://github.com/MDK8888/SageMode/assets/79173446/3680c0c8-4132-488e-aa07-2bf6561fc1d3)-click on the 'Create access key' button.
  14. On the next page, you will be given several options indicating what you are going to use your access key for. Select 'Local code' and then click on the 'Next' [button](https://github.com/MDK8888/SageMode/assets/79173446/7ddb6d5a-5086-46f6-a66a-9725263b83b7).
  15. The next page is optional - click on the orange 'Create access key' button in the bottom right.
  16. On the next page, you should get an option to download your access key and secret access key as a .csv [file](https://github.com/MDK8888/SageMode/assets/79173446/dd8abc01-3f6e-4abf-8b87-092e8b11ff67). Click on the 'Download .csv file' button.
  17. In the directory where you have sagemode installed, create a .env file and create the following:
      ```
      AWS_ACCESS_KEY_ID={access key ID from .csv file}
      AWS_SECRET_ACCESS_KEY={Secret access key from .csv file}
      AWS_REGION={your aws region}
      ```
Congratulations! We are now going to move away from users and move on to creating two roles, one which is responsible for using SageMaker, AWS's service for machine learning, and another for Lambda, AWS's service for serverless compute.

- Role Configuration:
  1. Navigate to the [IAM Dashboard](https://github.com/MDK8888/SageMode/assets/79173446/54d6f8c3-af7d-4187-856f-831b77b79f4f), and we're going to click on 'Roles' this time.
  2. Click on the orange 'Create role' button in the [top right](https://github.com/MDK8888/SageMode/assets/79173446/03cacbee-2387-4fc6-8a7f-aebc9dc23b02).
  3. You should now see a page with 'Select trusted entity' in the top left corner. Select the option 'AWS Service', and then on the bottom of the page there is a dropdown menu to choose the service or use case. Select 'SageMaker' from the dropdown menu or search for it. Click on the 'Next' [button](https://github.com/MDK8888/SageMode/assets/79173446/ead6b7f6-fa96-4019-b32d-f663a0b0f48d).
  4. On the next page, click on the orange 'Next' button. You should now see a page to enter a name for your role. Do that, scroll all the way down, and click on the orange 'Create role' button.
  5. You should now be back on the 'Roles' page. Click on the role that you just created, and then click on the 'Trust Relationships' tab. Recall your role arn from the previous section and copy and paste the below into the JSON:
     ```
     {
      "Version": "2012-10-17",
      "Statement": [
          {
              "Sid": "",
              "Effect": "Allow",
              "Principal": {
                  "Service": "sagemaker.amazonaws.com",
                  "AWS": "arn:aws:iam::..."
              },
              "Action": "sts:AssumeRole"
          }
        ]
      }
     ```
  5. Scroll down and click on the orange 'Update policy' button.
  6. You should now be back on the main page for your role. Click on 'Add permissions' and then click on 'Attach policies' this time.
  7. Search for the 'AutoScalingFullAccess' policy, select the checkbox next to it, and then click on the orange 'Add permissions' button.
  8. Finally, click on 'Add permissions', click on the 'Create inline policy' button.
  9. Between 'Visual' and 'JSON', click on 'JSON' and paste the following JSON in:
     ```
     {
          "Version": "2012-10-17",
          "Statement": [
              {
                  "Sid": "VisualEditor0",
                  "Effect": "Allow",
                  "Action": "iam:*",
                  "Resource": "*"
              }
          ]
     }
     ```
  10. Click on the orange 'Next' button in the bottom right.
  11. Come up with a name for your Policy, and then click on the orange 'Create policy' button.
  12. Congrats, we have made our SageMakerRole! To make our LambdaRole, we want to repeat steps 1-11, except we select 'AWSLambda_FullAccess' and 'AmazonSageMakerFullAccess', and add the inline policy for IAM in step 9.

Finally, once we have both the SageMakerRole and the LambdaRole, edit your .env file in the following way:
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...
SAGEMAKER_ROLE_ARN=arn:aws:iam::...rest of your SageMakerRole arn
LAMBDA_ROLE_ARN=arn:aws:iam::...rest of your LambdaRole arn
```
You are now ready to move on to tohe documentation section! Sincere apologies for the 'Quickstart' guide- it wasn't that quick, was it?

 # Documentation
 ![image](https://github.com/MDK8888/SageMode/assets/79173446/b3be1ce0-8fb8-4b0a-a729-c64afb348685)
 - Documentation will be coming very soon. However, in the meantime, check out the examples folder! To run any example, just create a python file in your virtual environment, copy and paste the example code in, and run it.

 # Roadmap
 - Deploy
   - add rapid teardown of EC2 and SageMaker Resources (0.1.1)
   - if needed, add HFEC2ResourceUser (0.1.1)
   - For EC2ResourceUsers, allow weights to be pulled from buckets (0.1.2)
   - allow for a "clean" deploy (all extra files created are also deleted) and a "dirty" deploy (no files created in the deployment process are deleted, better for debugging) (0.1.1)
   - Turn deployment chains into deployment graphs (0.1.2)
   - Add **kwargs to PytorchEC2ResourceUser (0.1.1)
 - Scale (1.x.x)
 - Monitor (2.x.x)
 - Ops/Other
   - Clean up the codebase and lint (0.1.1)
   - add CI/CD workflows
   - add pytest
   - add a cool sick icon (0.1.1)



 
