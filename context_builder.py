import pandas as pd
import numpy as np
import os
import sqlalchemy
import datetime

user_info = f"""
with sizelogs_frlr as(
	WITH sizelogs AS (
	  SELECT 
		user_id,
		height, 
		weight, 
		ROW_NUMBER() OVER(ORDER BY id ASC) AS record_id,  -- for the first record
		COUNT(*) OVER() as cnt                         -- for the last record
	  FROM senza_app_sizelogs
	  WHERE user_id = {{user_id}}
	)
	SELECT *, case when record_id = 1 then 'fr' else 'lr' end as record
	FROM sizelogs
	WHERE record_id = 1 OR record_id = cnt
)

select 
sau.first_name,
sau.female,
sau.birthdate,
DATE_PART('years',AGE(cast(NOW() AS date),sau.birthdate)) as age,
sau.macro_netcarbs,
sau.macro_protein,
sau.macro_fat,
sau.target_calories,
fr.height,
fr.weight as starting_weight,
coalesce(lr.weight,fr.weight) as current_weight,
sau.target_weight,
DATE_PART('day',NOW() - sau.created_at) as days_installed,
sau.days_under_net_carb_target,
sau.points_earned
from senza_app_users sau
left join sizelogs_frlr fr on sau.id = fr.user_id and fr.record = 'fr'
left join sizelogs_frlr lr on sau.id = lr.user_id and lr.record = 'lr'
where sau.id = {{user_id}}
"""

macros_info = f"""
select
  u.id as user_id,
  de.entry_timestamp::DATE,
  SUM(de.total_net_carbs) as net_carbs,
  SUM(de.total_proteins) as proteins,
  SUM(de.total_fats) as fats,
  SUM(de.total_calories) as cal,
  SUM(de.total_carbohydrates) as total_carbs
from senza_app_diaryentries de
	join senza_app_diaries d on de.diary_id=d.id
	join senza_app_fooditems f on f.nx_food_id = de.nx_food_id
	join senza_app_users u on u.id=d.user_id
where de.updated_at is not null
	and de.is_deleted = 'FALSE'
	and de.status = 'complete'
	and de.entry_timestamp is not null
	AND de.entry_timestamp > (NOW() - INTERVAL '10 days')
	AND u.id = {{user_id}}
GROUP BY de.entry_timestamp::DATE, u.id
ORDER BY entry_timestamp::DATE DESC
limit 15;
"""

gender_mapping = {True: 'Female', False: 'Male'}
    
def create_user_info_prompt(user_data):

    user = user_data["first_name"]
    height = int(user_data["height"])
    pronoun_mapping = {True: 'She', False: 'He'}
    gender_type_mapping = {True: 'woman', False: 'man'}

    pronoun = pronoun_mapping.get(user_data['female'], 'Not Specify')
    gender = gender_type_mapping.get(user_data['female'], 'Not Specify')


    if user_data.get("age"):
        age_in_years = int(user_data["age"])
    else:
        birthdate_str = user_data['birthdate']
        
        if birthdate_str is None:
            age_in_years = None
        else:
            #birthdate = datetime.datetime.strptime(birthdate_str, '%Y-%m-%d %H:%M:%S')
            #birthdate = datetime.datetime.strptime(birthdate_str, '%m/%d/%y %H:%M')
            
            try:
                birthdate = birthdate_str
            except ValueError:
                try:
                    birthdate = birthdate_str
                except ValueError:
                    print(f"Unable to parse date string {birthdate_str}")
                    birthdate = datetime.datetime.now()
                    
            
            today = datetime.datetime.now()
            age_in_years = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
            age_in_years = abs(age_in_years)
            age_in_years = user_data["age"]

        #age_in_years = abs((current_date - birthdate).days // 365)
    
    app_usage = create_time_in_app_prompt(user_data)
    #user_info = f'{user} is {age_in_years}-years-old {gender} ({height} inches tall) and is using ketosis for weight control - {app_usage}'    
    user_info = f'{user} has been using Senza for {app_usage} and is a {age_in_years}-years-old {gender} who is {height} inches tall.'

    #user_info = f'User info: User {user}, Gender {gender}, Age {age_in_years}, Height {height} inches.'
    return user_info

def create_string_table(user_data):
    # training_data["json_agg"][0]    
    table_string = ""
    user = user_data["first_name"]
    data_list = user_data["historical_macro_logs"]    
    if len(data_list) > 0:
        sorted_data_list = sorted(data_list, key=lambda x: x['entry_timestamp'])
        most_recent_date = sorted_data_list[-1]['entry_timestamp'] 
        for entry in sorted_data_list: 
            entry_date = entry['entry_timestamp']
            time_range_days = (most_recent_date - entry_date).days
            entry['time_range'] = f"last {time_range_days} days"
        
        n_logs = len(data_list)
        if n_logs < 10:
            table_string = f"{user} only logged {n_logs} day(s) of food items in the last 10 days:\n"
        else:
            table_string = f"{user} logged {n_logs} days of food items:\n"

        for entry in sorted_data_list[::-1]:
            time_range = entry['time_range']
            proteins = int(entry['proteins'])
            fats = int(entry['fats'])
            net_carbs = int(entry['net_carbs'])
            total_carbs = int(entry['total_carbs'])
            calories = int(entry['cal'])
            
            table_string += f"{time_range}: Proteins {proteins}g, Fats {fats}g, Net carbs {net_carbs}g, total carbs {total_carbs}g , {calories} calories \n"
    else:
        table_string = f"{user} logged no food items in the journal in the last 10 days \n"

    return table_string

def create_user_weight_prompt(user_data):
    pronoun_mapping = {True: 'Her', False: 'His'}
    gender_pronoun_mapping = {True: 'She', False: 'He'}

    pronoun = pronoun_mapping.get(user_data['female'], 'Not Specify')
    gender_pronoun = gender_pronoun_mapping.get(user_data['female'], 'Not Specify')
    # if the weight metrics are None will be setup to 0
    sw = round(user_data.get('starting_weight',0),1)
    cw = round(user_data.get('current_weight',0),1)
    tw = round(user_data.get('target_weight',0),1)
    
    #weight_string= f"{pronoun} initial weight was {sw} lbs,{pronoun} current weight is {cw} lbs,{pronoun} target weight is {tw} lbs."
    #weight_string = f"Weights: Starting weight {sw}, current weight {cw}, target weight {tw}, days under net carbs (NC) {unc}."
    weight_string = f"{pronoun} weight began at {sw} lbs, is currently {cw} lbs, and {gender_pronoun} wants to weight {tw} lbs"
    return weight_string


def create_time_in_app_prompt(user_data):
    days_installed = int(user_data["days_installed"])
    sub_prompt = f'{days_installed} day(s)'
    return sub_prompt


def create_macros_prompt(user_data):
    user = user_data["first_name"]
    pronoun_mapping = {True: 'She', False: 'He'}
    pronoun = pronoun_mapping.get(user_data['female'], 'They')

    proteins = user_data["macro_protein"]
    fats = user_data["macro_fat"]
    nc = user_data["macro_netcarbs"]
    cal = user_data["target_calories"]
    macros_string = f'{pronoun} has a goal to eat close to these macronutrient totals each day: Proteins {proteins}g , Fats {fats}g , net carbs {nc}g and {cal} calories'
    return macros_string


def create_context(user_data):
    user_info = create_user_info_prompt(user_data)
    weight_info = create_user_weight_prompt(user_data)
    additional_info = create_string_table(user_data)
    macros_info = create_macros_prompt(user_data)

    return f'{user_info}\n{weight_info}\n{macros_info}\n{additional_info}'

def format_text(user_data):
    # gender_mapping = {'F': 'Female', 'M': 'Male'}

    user_context = create_context()

    system_message = '<|SYSTEM|># Nutrition Assistant: Answer questions related to Senza nutrition app, using the context provided within triple backticks.\nIf a question is unrelated to the app, respond: I am sorry, I\'m afraid I cannot answer that question.'
    return system_message + f"\n```\n{user_context}```\n<|USER|>{question}<|ASSISTANT|>"


def format_output(instance):
    output_text = instance["text"]
    formatted_openai = f" {output_text}{EOS_TOKEN}"
    return formatted_openai

def preprocess_function(sample, tokenizer, max_source_length, max_target_length, padding="max_length"):
    # add prefix to the input for t5
    inputs = sample["input"]
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=max_target_length, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(lbl if lbl != tokenizer.pad_token_id else -100) for lbl in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

    
def build_user_context(user_id):

	conn_template = f'postgresql+psycopg2://{{user}}:{{password}}@{{host}}:{{port}}/{{dbname}}'

	conn_string = conn_template.format(
		dbname="d1fh04q4apocim",
		user="u8qo5htq3eaqh2",
		password=os.getenv("DB_PASS"),
		host="ec2-52-21-86-209.compute-1.amazonaws.com",
		port="5432"
	)
	
	engine = sqlalchemy.create_engine(conn_string)

	with engine.connect() as conn:
		user_general_info = pd.read_sql_query(sqlalchemy.text(user_info.format(user_id=user_id)),conn)
		user_macros_info = pd.read_sql_query(sqlalchemy.text(macros_info.format(user_id=user_id)),conn)
		
	user_data = user_general_info.to_dict(orient="records")[0]
	user_data["historical_macro_logs"] = user_macros_info.to_dict(orient="records")
	context = create_context(user_data)

	return context