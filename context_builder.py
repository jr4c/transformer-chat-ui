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
		COUNT(*) OVER() as cnt,                         -- for the last record
		created_at
	  FROM senza_app_sizelogs
	  WHERE user_id = {{user_id}}
	)
	SELECT *, case when record_id = 1 then 'fr' else 'lr' end as record
	FROM sizelogs
	WHERE record_id = 1 OR record_id = cnt
)

select
cast(NOW() AS date) as now_date,
sau.id as user_id,
sau.first_name,
sau.female,
CASE WHEN sau.female THEN 'female' else 'male' END as  gender,
sau.birthdate,
cast(DATE_PART('years',AGE(cast(NOW() AS date),sau.birthdate)) as INTEGER) as age,
sau.device_type,
sau.macro_netcarbs,
sau.macro_protein,
sau.macro_fat,
sau.target_calories,
COALESCE(fr.height,0) as height,
COALESCE(fr.weight,0) as starting_weight,
coalesce(lr.weight,COALESCE(fr.weight,0)) as current_weight,
COALESCE(sau.target_weight,0) as target_weight,
DATE_PART('day',cast(NOW() AS date) - sau.created_at) as days_installed,
DATE_PART('day',cast(NOW() AS date) - lr.created_at) as days_last_wlog,
sau.days_under_net_carb_target,
sau.points_earned
from senza_app_users sau
left join sizelogs_frlr fr on sau.id = fr.user_id and fr.record = 'fr'
left join sizelogs_frlr lr on sau.id = lr.user_id and lr.record = 'lr'
where sau.id = {{user_id}}
"""

diaries_info = f"""
select
cast(NOW() AS date) as now_date,
sad.user_id,
sade.*
from public.senza_app_diaryentries sade
left join public.senza_app_diaries sad on sad.id = sade.diary_id
where sade.is_deleted = false
and sad.user_id in ({{user}})
AND sade.entry_timestamp > (NOW() - INTERVAL '90 days')
"""

dow = { 0:"Monday", 1: "Tuesday", 2: "Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday" }

meal_times = {
    "12 AM": "Late Night Snack",
    "1 AM": "Late Night Snack",
    "2 AM": "Late Night Snack",
    "3 AM": "Late Night Snack",
    "4 AM": "Late Night Snack",
    "5 AM": "Breakfast",
    "6 AM": "Breakfast",
    "7 AM": "Breakfast",
    "8 AM": "Breakfast",
    "9 AM": "Breakfast",
    "10 AM": "Mid-Morning Snack",
    "11 AM": "Mid-Morning Snack",
    "12 PM": "Lunch",
    "1 PM": "Lunch",
    "2 PM": "Lunch",
    "3 PM": "Afternoon Snack",
    "4 PM": "Afternoon Snack",
    "5 PM": "Afternoon Snack",
    "6 PM": "Dinner",
    "7 PM": "Dinner",
    "8 PM": "Dinner",
    "9 PM": "Late Night Snack",
    "10 PM": "Late Night Snack",
    "11 PM": "Late Night Snack"
}

sort_meal_time = {
    "Breakfast": 0,
    "Mid-Morning Snack": 1,
    "Lunch": 2,
    "Afternoon Snack": 3,
    "Dinner": 4,
    "Late Night Snack": 5,
}


target_columns = [
    "user_id",
    "entry_timestamp",
    "date",
    "dayofweek",
    "hour",
    "time_12_hour",
    "name",
    "amount_value",
    "amount_unit",
    "total_proteins",
    "total_fats",
    "total_net_carbs",
    "total_calories",
    "now_date",
]


sort_meal_12hours = {
    "12 AM": 0,
    "1 AM": 1,
    "2 AM": 2,
    "3 AM": 3,
    "4 AM": 4,
    "5 AM": 5,
    "6 AM": 6,
    "7 AM": 7,
    "8 AM": 8,
    "9 AM": 9,
    "10 AM": 10,
    "11 AM": 11,
    "12 PM": 12,
    "1 PM": 13,
    "2 PM": 14,
    "3 PM": 15,
    "4 PM": 16,
    "5 PM": 17,
    "6 PM": 18,
    "7 PM": 19,
    "8 PM": 20,
    "9 PM": 21,
    "10 PM": 22,
    "11 PM": 23
}


#######################################PARSERS############################################


meal_history_template = f"""{{meal_times}}:
{{food_item}}
- Total: {{total_proteins}}g proteins, {{total_fats}}g fats, {{total_net_carbs}}g net carbs, {{total_calories}} calories
"""

def execute_format(meal):
    return meal_history_template.format(**meal)
###########################################################################################
meal_current_template = f"""- {{meal_times}}:

Macronutrients requirement:
{{total_proteins}}g proteins, {{total_fats}}g fats, {{total_net_carbs}}g net carbs and {{total_calories}} calories
### RECOMMENDATION
{{food_item}}
"""

def format_response(meal):
    return meal_current_template.format(**meal)

##########################################################################################
question_current_template = f"""\
Recommend a meal for today {{time_12_hour}}, taking into consideration the User Information, Meal history and the following macronutrient requirements:\nat or above {{total_proteins}}g protein, at or below {{total_fats}}g fat, below {{total_net_carbs}}g net carbs and around {{total_calories}} calories
"""

def generate_question(meal):

    return question_current_template.format(**meal)

##########################################################################################

user_information_text = f"""\
{{age}}-year-old {{gender}} with a current weight of {{current_weight}} and a target weight of {{target_weight}}
Daily macronutrient targets: at or above {{macro_protein}}g protein, at or below {{macro_fat}}g fat, below {{macro_netcarbs}}g net carbs and {{target_calories}} total calories\
"""

def generate_user_info(info):
    return user_information_text.format(**info)

####################################PROMPT#TEMPLATE#######################################

prompt_template=f"""### Context
User Information:

{{user_context}}

Meal history:

{{meal_history}}
### Instruction:
{{task}}
### Recommendation:
"""

##########################################################################################

meal_ = f"""{{time_12_hour}}
{{food_item}}
- Total: {{total_proteins}}g protein, {{total_fats}}g fat, {{total_net_carbs}}g net carbs, {{total_calories}} calories
"""
def xg(meal):
    return meal_.format(**meal)

##########################################################################################
day_ = f"""{{time_window}}
{{history}}
"""

def xf(t,meal_history):
    return day_.format(time_window=t,history=meal_history)

##########################################################################################

def round_sum(i):
    v = sum([int(round(x,0)) for x in i.head(5)])
    return v

def bucket_dates(date_series):
    days = list([-i.days for i in date_series.values])
    bucketed_dates = pd.cut(
        days,  # Converts timedelta to positive days
        bins=[-np.inf,0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,np.inf],
        labels=[
            'Today',
            'Yesterday',
            '2 days ago',
            '3 days ago',
            '4 days ago',
            '5 days ago',
            '6 days ago',
            '7 days ago',
            '8 days ago',
            '9 days ago',
            '10 days ago',
            'More than 10 days'
        ],
        include_lowest=True,
        ordered=False
    )

    return bucketed_dates.astype(str)

def concatenate_food(f):
    return '\n'.join(f.head(5))

def get_fooditem_text(instance):
    q = instance['amount_value']
    u = str(instance['amount_unit'])
    n = instance["name"]

    proteins = int(round(instance["total_proteins"],0))
    fats = int(round(instance["total_fats"],0))
    nc = int(round(instance["total_net_carbs"],0))
    cal = int(round(instance["total_calories"],0))

    # Verify if is a integer
    if int(q) == float(q):
        q = int(q)

    q = round(q,4)

    #Default response is with units unless the unit is in the Food name
    response = f"- {n} ({q} {u}): {proteins}g protein, {fats}g fat, {nc}g net carbs, {cal} calories"
    if u in n.lower():
        response = f"- {q} {n}: {proteins}g protein, {fats}g fat, {nc}g net carbs, {cal} calories"
    return response

def build_user_context(user_id,meal_time_filter):
    conn_template = f'postgresql+psycopg2://{{user}}:{{password}}@{{host}}:{{port}}/{{dbname}}'
    conn_string = conn_template.format(
		dbname="d1fh04q4apocim",
		user="u8qo5htq3eaqh2",
		password=os.getenv("DB_PASS"),
		host="ec2-52-21-86-209.compute-1.amazonaws.com",
		port="5432"
	)
    user_info_dict = {}
    engine = sqlalchemy.create_engine(conn_string)
    with engine.connect() as conn:
        df = pd.read_sql_query(sqlalchemy.text(diaries_info.format(user=user_id)),conn)
        info = pd.read_sql_query(sqlalchemy.text(user_info.format(user_id=user_id)),conn)        

    if df.shape[0]==0:
            prompt = "No Data"
    else:
        df['dayofweek'] = [dow.get(d) for d in df['entry_timestamp'].dt.dayofweek]
        df['hour'] = [ int(i) for i in df['entry_timestamp'].dt.hour]
        df['time_12_hour'] = df['entry_timestamp'].dt.strftime('%I %p').str.lstrip('0')
        df["date"] = df['entry_timestamp'].dt.date
        df = df.sort_values(by=["user_id",'entry_timestamp'])
        df = df[target_columns]
        df["food_item"] = df.apply(get_fooditem_text,axis=1)
        df["meal_times"] = df.apply(lambda i: meal_times.get(i["time_12_hour"]),axis=1)
        df["sort"] = df.apply(lambda i: sort_meal_time.get(i["meal_times"]),axis=1)
        df["sort_12hours"] = df.apply(lambda i: sort_meal_12hours.get(i["time_12_hour"]),axis=1)


        data = df.sort_values(
            by=["total_calories","hour"]
            ,ascending=[False,True]
        ).groupby(["user_id", "date","now_date", "dayofweek","sort","meal_times","time_12_hour"]).agg({
            "food_item": concatenate_food,
            "total_proteins":round_sum,
            "total_fats":round_sum,
            "total_net_carbs":round_sum,
            "total_calories":round_sum,
        })


        data = data.reset_index()

        prompt_response = []
        max_n = 3*3
        for user_id , user_dairy in data.groupby("user_id"):
            df = user_dairy
            sample = df.iloc[:].copy()
            sample.index = sample.reset_index(drop=True).index[::-1]


            diff_dates = sample["date"]-sample["now_date"]
            sample["time_window"] = bucket_dates(diff_dates)

            history = sample
            history = history.query("time_window != 'More than 10 days'")
            
            # just in case
            history = history.loc[max_n:0]

            main_task = {
                "time_12_hour":f"{meal_time_filter}",
                "total_proteins":"XX",
                "total_fats":"YY",
                "total_net_carbs":"ZZ",
                "total_calories":"WW"
            }

            info_value = generate_user_info(info.to_dict(orient="records")[0])
            
            #NO INFO THEN
            if history.shape[0] > 0:

                logged_days = []
                for (d,time_window), meals in history.groupby(["date","time_window"]):
                    meal_history = '\n'.join([xg(meal) for meal in meals.to_dict(orient="records")])
                    logged_days.append(xf(time_window,meal_history))
                

                history_log = ''.join(logged_days)
                
                prompt = prompt_template.format(
                    user_context = info_value,
                    meal_history = history_log,
                    task = generate_question(main_task)
                )
            else:
                meal_history = f"No recent logs"
                prompt = prompt_template.format(
                        user_context = info_value,
                        meal_history = meal_history,
                        task = generate_question(main_task)
                )

    context = prompt

    return context