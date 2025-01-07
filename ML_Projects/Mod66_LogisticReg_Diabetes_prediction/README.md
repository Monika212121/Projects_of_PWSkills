This is my second ML End-to-end project.

## Business Problem : Diabetes prediction.

It uses Logistic Regression model for Output prediction.



## Afterwards for deployment, create '.ebextentions' named folder and 'python.config' file inside this folder.

Also make 2 changes:

* 1.) Change name of 'app.py' file to 'application.py' as while deploying, it is by-default name.

* 2.) In application.py file now, rename 'app' to 'application' and assign this 'application' variable to new variable 'app' .