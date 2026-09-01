# Part 1: Warmup — Cloud Concepts
## Cloud Concepts Question 1
**What is the core economic model of cloud computing, and how does it differ from owning your own servers?**
* the core economic model of cloud computing is getting computer resources from a provider like storage, processing power, networking, and paying for what you use. It differs from owning your own server because you don't have to buy and maintain your own computer services yourself.

## Cloud Concepts Question 2
**What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.**
* Vertical scaling is upgrading the computer or hardware, while horizontal scaling is adding more computers or hardware. Vertical scaling for more machine power (RAM, CPU, GPU), Horizontal scaling for splitting work so that you could process faster without overloading. 
**Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.**

**A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.**
* Horizontal scaling. adding more machines helps with the workload of handling more users.
**A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.**
* Vertical Scaling. A single machine learning training process typically needs shared GPU memory and RAM on a single host machine to train faster without the complexity of distributed training.
**A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.**
* Horizontal Scaling. . Because the file processing work is parallelizable and can be split across multiple worker nodes, distributing the jobs across multiple instances processes the batch much faster.

## Cloud Concepts Question 3

**Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.**
Gmail: SaaS. open a browser to use the app. 
Azure Virtual Machines: IaaS. It provides raw virtualized compute infrastructure
AWS S3 (Simple Storage Service): IaaS. It provides raw, scalable object storage infrastructure where you manage the data and access policies, but AWS manages the physical hardware.
GitHub Codespaces: PaaS. It provides a pre-configured, cloud-hosted development platform and runtime environment where you focus solely on writing and executing code.
Snowflake: SaaS. It is fully managed data warehouse software where the provider handles all infrastructure, tuning, and maintenance behind a query interface.
Supabase: BaaS. It provides ready-to-use backend services (PostgreSQL database, authentication, edge functions, and storage) directly via APIs so you don't have to build custom server backends.
Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.
* IaaS is infrastructure. Raw compute, storage, network, You set up everything else. Most flexible, most work.
    * Example: Azure Virtual Machines (or AWS S3 / EC2).
    * Responsibilty: : configuring and managing the operating system, security patches, network configurations, database installations, runtimes, and all application code.
* PaaS: Provider manages infrastructure, you bring your own code.
    * Example: GitHub Codespaces (or Heroku / AWS Elastic Beanstalk).
    * Responsibility: writing, deploying, and maintaining your application code, dependencies, and application-level data.
* SaaS: A complete, centrally hosted application delivered over the internet, typically accessed via a web browser or API.
    * Gmail
    * esponsibility: You are only responsible for configuring user settings, managing your own data/content, and managing access permissions.

## Cloud Concepts Question 4

**What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?**
* a managed data platform pre-wire the pieces for you, optimizing specifically for data and analytics workloads. it's not a separate cloud, it's a curated layer that provisions and manages cloud resources on your behalf.
    * What you gain: Fast time-to-value, zero cluster management/tuning overhead, built-in optimization, and seamless data sharing.
    * What you give up: Fine-grained infrastructure control, cost efficiency at high volume (you pay a software margin on top of raw cloud compute), and you risk vendor lock-in.

## Cloud Concepts Question 5

**The lesson names two situations where the cloud is probably not the right choice. What are they?**
* If your dataset fits comfortably on a single machine and you do not have massive compute demands. 


# Part 2: Warmup — Cloud Landscape
## Cloud Landscape Question 1

**Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.**
* AWS: It's the biggest and oldest with the most services, so startups and giant enterprises use it for almost everything.
* Microsoft Azure: It connects perfectly with Microsoft enterprise software, making it the top choice for big corporate businesses.
* Google Cloud (GCP): It has the best data and machine learning tools, so tech-heavy companies and data scientists prefer it.

## Cloud Landscape Question 2

**The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.**
* Access. Supabase accounts are self-provisioned at supabase.com in under two minutes and the free tier is sufficient for everything in this course.
* Pedagogical fit. Supabase stores data as rows and columns in a relational database. A relational database is more transferable: querying, filtering, and reasoning about structured data is a skill you'll use in almost every data role. 
* Pipeline coherence. The ETL pipeline you build in weeks 9–11 has a raw zone and an enriched zone

**Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?**
* You should use the right cloud tool for the job. It doesn't have to be overly complicated. If it gets the job done, then it gets the job done. 

## Cloud Landscape Question 3

**For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.**

**You need to store 10 TB of image files and retrieve them by filename from any machine.**
    * Object storage; AWS S3.
**You need to run an ML training job on a GPU for four hours, then shut it down.**
    * Virtual machines (or raw compute); AWS EC2.
**You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.**
    * Serverless compute; Google Cloud Run.
**You need to send structured data to a large language model and get a text response back.**
    * LLM API; OpenAI API.

## Cloud Landscape Question 4

**The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?**

* I want to build a website that tracks my workouts and writes a weekly summary for me, using Supabase for the database and the OpenAI API to generate the text summaries.
* Consolidating to one provider makes billing and security way easier to manage, but you give up the ability to use the absolute best tool for the job.