This is my first ML End to end project.

It uses Regression model for Output prediction.

# If there is any git initiated already, then check with this command.
```
    git remote -v 

```

# If there is any git initiated already, then remove the existing origin with this command.
```
    git remote remove origin

```

# Git commands used to setup this repository  and sync with the online github.
```
    git init
    git add .
    git status
    git add .
    git commit -m "first commit in ML"
    git config --global user.email "mgadewar12@gmail.com"
    git config --global user.name "Monika212121"
    git commit -m "first commit in ML"
    git branch -M main
    git branch
    git remote add origin https://github.com/Monika212121/Mod65_ML_Forest_fire.git
    git push -u origin main
  
  ```

  Afterwards for deployment, create .ebextentions folder and pyhton.config file inside it.

  ### Also make 2 changes:-

  * 1.) Change name od app.py to application.py as while deploying, it is by default name.
  * 2.) In application.py file now, rename 'app' to 'application' and assign this 'application' variable to new variable 'app' .