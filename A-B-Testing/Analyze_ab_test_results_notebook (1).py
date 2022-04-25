#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv("ab_data.csv")
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


index = df.index
number_of_rows = len(index)
print("Number of rows = ",number_of_rows)


# In[4]:


df.shape[0]


# In[5]:


df.info()


# c. The number of unique users in the dataset.

# In[6]:


df["user_id"].nunique()


# d. The proportion of users converted.

# In[7]:


## user converted / all users


# In[8]:


df.query('converted == "1"').shape[0] / number_of_rows


# e. The number of times the `new_page` and `treatment` don't line up.

# In[9]:


df["group"].unique()


# In[10]:


df["landing_page"].unique()


# In[11]:


old_page_treatment = df.query('group == "treatment" and landing_page == "old_page"').shape[0]
new_page_control = df.query('group == "control" and landing_page == "new_page"').shape[0]
nums_of_times = old_page_treatment + new_page_control
print(nums_of_times)


# In[12]:


## Another way


# In[13]:


old_page_treatment = df[(df["group"] == "treatment") & (df["landing_page"] == "old_page")].shape[0]
new_page_control = df[(df["group"] == "control") & (df["landing_page"] == "new_page")].shape[0]
nums_of_times = old_page_treatment + new_page_control
print(nums_of_times)


# f. Do any of the rows have missing values?

# In[14]:


df.isnull().any().sum()


# ### The dataset is clean it has no nulls

# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[15]:


new_page_treatment = df[(df["group"] == "treatment") & (df["landing_page"] == "new_page")]
old_page_control = df[(df["group"] == "control") & (df["landing_page"] == "old_page")]
df2 = new_page_treatment.append(old_page_control, ignore_index=True)


# In[16]:


df2.info()


# #### Another way

# In[17]:


old_page_treatment = df.query('group == "treatment" and landing_page == "new_page"')
new_page_control = df.query('group == "control" and landing_page == "old_page"')
df2 = old_page_treatment.append(new_page_control, ignore_index=True)


# In[18]:


df2.info()


# In[19]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[20]:


df2["user_id"].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[21]:


df2.head()


# In[22]:


## That's it


# In[23]:


df2[df2.user_id.duplicated()]


# c. What is the row information for the repeat **user_id**? 

# In[24]:


df.query('user_id == "773192"')


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[25]:


df2.drop(2893,inplace=True)


# In[26]:


df2.info()


# ### Done

# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[27]:


df2["converted"].mean()


# In[28]:


df2.query('converted == "1"').shape[0] / df2.shape[0]


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[29]:


df2[df2["group"] == "control"]["converted"].mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[30]:


df2[df2["group"] == "treatment"]["converted"].mean()


# d. What is the probability that an individual received the new page?

# In[31]:


df2[df2["landing_page"] == "new_page"].shape[0] / df2.shape[0]


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# **Your answer goes here.**

# **No, it does not seem that the new_page leads to more conversions. the probability of treatment group = 11.8% and control group = 12%
# the difference 0.02% which appears to be negligible.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Put your answer here.**

# $$ H_{0} : P_{old}  = P_{new}$$
# $$ H_{1} : P_{new}  > P_{old}$$
# 
# **Or**
# 
# $$ H_{0} : P_{old}  - P_{new} = 0 $$
# $$ H_{1} : P_{new}  - P_{old} > 0 $$

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[32]:


P_null = df2["converted"].mean()
print(P_null)


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[33]:


df2.converted.mean()


# c. What is $n_{new}$?

# In[34]:


n_new = df2[df2["landing_page"] == "new_page"].shape[0]
print(n_new)


# d. What is $n_{old}$?

# In[35]:


n_old = df2[df2["landing_page"] == "old_page"].shape[0]
print(n_old)


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[36]:


new_page_converted = np.random.binomial(1, P_null, n_new)


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[37]:


old_page_converted = np.random.binomial(1, P_null, n_old)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[38]:


diff = new_page_converted.mean() - old_page_converted.mean()
print(diff)


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[40]:


p_diffs = []
new_converted_simulation = np.random.binomial(n_new, P_null, 10000)/n_new
old_converted_simulation = np.random.binomial(n_old, P_null, 10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation


# In[ ]:


null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)
plt.hist(null_vals)
plt.axvline(x=diff, color='r')


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[42]:


plt.hist(p_diffs);


# In[ ]:


p_diffs = np.array(p_diffs)
plt.hist(p_diffs)
# plot line for observed statistic
plt.axvline(x=diff,color="red");


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[70]:


obs_diff = df2['converted'][df2['group'] == 'treatment'].mean() - df2['converted'][df2['group'] == 'control'].mean()


# In[71]:


print(obs_diff)


# In[72]:


len(p_diffs[p_diffs > obs_diff])/len(p_diffs)


# In[73]:


low_probability = (p_diffs < obs_diff).mean()
high_probability = (p_diffs.mean() + (p_diffs.mean() - obs_diff) < p_diffs).mean()


# In[74]:


print('High',high_probability)
print('Low',low_probability)


# In[75]:


plt.hist(p_diffs);
plt.axvline(obs_diff, color='red');
plt.axvline(p_diffs.mean() + (p_diffs.mean() - obs_diff), color='red');


# In[48]:


p_value = low_probability + high_probability
print(p_value)


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Put your answer here.**

# - **If the P-value were under 0.05,then it would be a very low probability ,if the null hypothesis were true, finding a value equal to or greater or lesser than obs_diff.**
#    
# **However,the P-value is above 0.05, which means i do not have evidence to reject the null hypothesis.** $$ H_{0} : P_{old}  = P_{new}$$

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[49]:


get_ipython().system('pip install statsmodels')


# In[50]:


import statsmodels.api as sm

convert_old = df2.query("landing_page == 'old_page' and converted == '1'").shape[0]
convert_new = df2.query("landing_page == 'new_page' and converted == '1'").shape[0]
n_old = df2.query('group == "control"').shape[0]
n_new = df2.query('group == "treatment"').shape[0]
print(convert_old,convert_new,n_old,n_new)


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[76]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new],alternative='smaller')
print(z_score, p_value)


# In[52]:


from scipy.stats import norm

norm.cdf(1.3109241984234394), norm.ppf(1-(0.05))


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Put your answer here.**

# **The z-score of 1.311 does not surpasses the critical value of 1.644, and the P-value is 0.905 so we cannot reject the null hypothesis that the difference between the two proportions is no different from zero.**
# 
# - **Also, the p-value here is 0.1899, which isn't underneath our alpha of 0.05. This p-value is comparable to the past p-value of 0.1998, so the z-test shows up to agree with the past findings.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Put your answer here.**

# **The logistic regression, which can decide conversion or not.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[53]:


df2['intercept'] = 1
df2[['new_page','old_page']] = pd.get_dummies(df2['landing_page'])
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[54]:


logit_model = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
predictions = logit_model.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[55]:


predictions.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **Put your answer here.**

# -  The p-value of ab_page is 0.19 that's means it is not significant in predicting whether the individual converts or not.
# -  This model is attempting to predict whether a user will convert depending on their page.
# -  The null hypothesis when ab_page = 1, converted = 0; 
# -  The alternative hypothesis  when ab_page = 1, converted almost to be 1.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Put your answer here.**

# -  I think it will affect on results because we used two features only ,we donot use group feature ,it would affect on the result 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[56]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# In[57]:


df_new.head()


# In[58]:


df_new['country'].unique()


# In[59]:


### Create the necessary dummy variables
df_new[['US', 'UK', 'CA']] = pd.get_dummies(df_new['country'])


# In[60]:


df_new.head()


# In[61]:


logit_model_2 = sm.Logit(df_new['converted'], df_new[['intercept', 'ab_page',"UK","CA"]])
pred = logit_model_2.fit()


# In[62]:


pred.summary()


# **Based on the p-values above, it also does not appear as though country has a significant impact on conversion.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[63]:


## Add new features 
df_new['US_ab_page'] = df_new['US'] * df_new['ab_page']
df_new['CA_ab_page'] = df_new['CA'] * df_new['ab_page']


# In[64]:


df_new.head()


# In[65]:


### Fit Your Linear Model And Obtain the Results
## fit the model with the new columns
logit_model_3 = sm.Logit(df_new['converted'], df_new[['intercept',"US_ab_page","CA_ab_page"]])
result = logit_model_3.fit()


# In[66]:


result.summary()


# In[68]:


np.exp(result.params)


# In[69]:


print(1/0.919292)


# **Based on the result above, one p-value does present as statistically significant: the interaction of US and ab_page (p = 0.026; p < 0.05).**

# ### Conclusions

# **p-value (ab_page) = .19 > 0.05 .
# There is no significant interaction between 'ab_page' and 'country' because the coefficient of 'ab_page' (-0.015) does not change as 'country' is introduced.
# Ultimately, we do not have enough evidence to reject the null hypothesis based on any of our A/B testing.**

# **The result tells us an interpretation for the US_page coefficient. Holding all other variables constant, a user from US who gets the new page would be about 1.087 times more likely to convert.**

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by beginning the next module in the program.

# In[ ]:




