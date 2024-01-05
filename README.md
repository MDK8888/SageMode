# SageMode
SageMode is a python library for deploying, scaling, and monitoring machine learning models and LLMs in particular at scale. It is native to AWS, which means that SageMode uses boto3 under the hood to interact with services like EC2, S3, SageMaker, and Lambda.

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



 
