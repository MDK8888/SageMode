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
- Role Configuration: You now need to create a .env file in the environment where you have installed sagemode.
  1. Create an AWS Account: Go to https://aws.amazon.com/console/ and create a root account and follow the steps as necessary.
  2. Log in to the AWS console using your root account.
  3. In the search bar on the home page, enter IAM and click on the first link:
     ![image](https://github.com/MDK8888/SageMode/assets/79173446/4ce9e651-6a54-494d-b4b9-09c197e59dbb)
  4. Click on Users:
     ![image](https://github.com/MDK8888/SageMode/assets/79173446/370dcca0-9b8e-4f01-8f98-c3d6f45c5778)
  5. Click on create user.
     ![image](https://github.com/MDK8888/SageMode/assets/79173446/1666a0e1-34bf-47dc-b687-60cdcdad9d22)
  6. Enter a name for your user, and then click on 'Next'.
  7. In the permissions options, search for the following policies which are labeled 'AWS Managed' in the search bar and check the box next to each one when they appear. We will go back and add the inline policy later ourselves.
     ![image](https://github.com/MDK8888/SageMode/assets/79173446/d724743c-324e-4354-9c12-92ea25217c10)
  8. Now, when you go back to the users page and click on the IAM User you just created, you can now see something called the arn. Keep this in mind-it will be important in the next step.
     ![image](https://github.com/MDK8888/SageMode/assets/79173446/de4a65af-d4f8-4f0e-8198-7df86118a940)
  10. Go back to your user, and choose to 'add permissions', and then 'create inline policy'. Between visual and JSON, select JSON and something like this should appear:
      ![image](https://github.com/MDK8888/SageMode/assets/79173446/3053ef66-0c47-4492-aeb3-89eee4916451)
  11. Copy and paste the following JSON, replacing it with your arn:
      ```
      {
     	"Version": "2012-10-17",
     	"Statement": [
     		{
     			"Sid": "Statement1",
     			"Effect": "Allow",
     			"Action": [
     				"iam:GetRole",
     				"iam:PassRole"
     			],
     			"Resource": [
     				<<your arn as a string>>
     			]
     		}
     	]
     }
      ```
  (Will come back and finish this later today)
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



 
