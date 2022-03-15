from collections import defaultdict
from io import StringIO
import json
import boto3
import pandas as pd
import s3_helper

BUCKET_NAME = 'virtual-regatta'
ROOT_FOLDER = 'logs_boat_status/'
temp_json = './src/user_race_data_2021-11-02_4.json'


def store_user_data():
    data = defaultdict()
    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    user_folders = s3_helper.list_folders(BUCKET_NAME, ROOT_FOLDER, client)
    for i, u_folder in enumerate(user_folders):
        user_id = u_folder.split('/')[-2]
        print(f'user #{i}/{len(user_folders)}')
        game_folders = s3_helper.list_folders(BUCKET_NAME, u_folder, client)
        for g_folder in game_folders:
            race_id = g_folder.split('/')[-2]
            key = f'{g_folder}user_race_data.csv'
            if s3_helper.key_exists(BUCKET_NAME, key, s3):
                content = s3_helper.read_concat_csv_boat_status(
                    BUCKET_NAME, key, s3)
                if user_id not in data:
                    data[user_id] = defaultdict()
                if race_id not in data[user_id]:
                    data[user_id][race_id] = defaultdict()
                data[user_id][race_id]['number_records'] = len(
                    content) - 1  # Header
                data[user_id][race_id]['last_calc'] = content[-1][0]

    with open(temp_json, 'w+') as json_file:
        json.dump(data, json_file)


with open('./src/followed.json') as json_file:
    FOLLOWED = json.load(json_file)['data']


def remove_players_missing_data():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    client = boto3.client('s3')
    user_folders = s3_helper.list_folders(BUCKET_NAME, ROOT_FOLDER, client)
    for u_folder in user_folders:
        user_id = u_folder.split('/')[-2]
        print(f'user {user_id}')
        game_folders = s3_helper.list_folders(BUCKET_NAME, u_folder, client)
        for g_folder in game_folders:
            race_id = g_folder.split('/')[-2]
            print(f'race {race_id}')
            key = f'{g_folder}user_race_data.csv'
            if not s3_helper.key_exists(BUCKET_NAME, key, s3):
                print(f'Deleting {g_folder}')
                bucket.objects.filter(Prefix=g_folder).delete()


def remove_unwanted_players():
    with open(temp_json, 'r') as json_file:
        user_race_data = json.load(json_file)
        remove = False
        remove_cnt = 0
        total_cnt = 0
        not_followed_cnt = 0
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(BUCKET_NAME)
        for user, value in user_race_data.items():
            total_cnt += 1
            for race, data in value.items():
                if data['number_records'] < 100:
                    print(
                        f'User {user} has less than 100 records for race {race}')
                    remove = True
            if user not in FOLLOWED:
                not_followed_cnt += 1
                print(f'User {user} is NOT followed, deleting it')
                remove = True
            else:
                print(f'User {user} is followed, NOT deleting it')
                remove = False
            if remove:
                remove_cnt += 1
                key = f'{ROOT_FOLDER}{user}/{race}/'
                if not s3_helper.folder_exists(s3, BUCKET_NAME, key):
                    print(f'Could not find user {user}')
                else:
                    bucket.objects.filter(Prefix=key).delete()
                    print(f'Deleted {key}')
                remove = False
        print(total_cnt)


def merge_legacy_files():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    client = boto3.client('s3')
    user_folders = s3_helper.list_folders(BUCKET_NAME, ROOT_FOLDER, client)
    for u_folder in user_folders:
        user_id = u_folder.split('/')[-2]
        print(f'user {user_id}')
        game_folders = s3_helper.list_folders(BUCKET_NAME, u_folder, client)
        for g_folder in game_folders:
            race_id = g_folder.split('/')[-2]
            print(f'race {race_id}')
            legacy_folders = s3_helper.list_folders(
                BUCKET_NAME, g_folder, client)
            l_content = []
            for j, l_folder in enumerate(legacy_folders):
                print(f"Folder {j}/{len(legacy_folders)}")
                l_folder_name = l_folder.split('/')[-2]
                if 'leg' in l_folder_name:
                    continue
                l_files = s3_helper.get_all_files(
                    BUCKET_NAME, l_folder, s3_client=client)
                for i, l_file in enumerate(l_files):
                    print(f"File {i}/{len(l_files)}")
                    l_content.append(s3_helper.read_csv_boat_status(
                        BUCKET_NAME, l_file['Key'], s3))
            if l_content != []:
                csv_buffer = StringIO()
                key = f'{g_folder}user_race_data.csv'
                content = s3_helper.read_concat_csv_boat_status(
                    BUCKET_NAME, key, s3)
                content_df = pd.DataFrame(content)
                content_df.columns = content_df.iloc[0]
                content_df.drop(content_df.index[0], inplace=True)

                l_content_df = pd.DataFrame(l_content)
                l_content_df.columns = content_df.columns

                appended_df = content_df.append(l_content_df)
                appended_df.to_csv(csv_buffer, header=True, index=False)

                s3.Object(BUCKET_NAME, key).put(
                    Body=csv_buffer.getvalue())


def remove_legacy_files():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    client = boto3.client('s3')
    user_folders = s3_helper.list_folders(BUCKET_NAME, ROOT_FOLDER, client)
    for u_folder in user_folders:
        user_id = u_folder.split('/')[-2]
        print(f'user {user_id}')
        game_folders = s3_helper.list_folders(BUCKET_NAME, u_folder, client)
        for g_folder in game_folders:
            race_id = g_folder.split('/')[-2]
            print(f'race {race_id}')
            legacy_folders = s3_helper.list_folders(
                BUCKET_NAME, g_folder, client)
            for j, l_folder in enumerate(legacy_folders):
                print(f"Folder {j}/{len(legacy_folders)}")
                l_folder_name = l_folder.split('/')[-2]
                if 'leg' in l_folder_name:
                    continue
                bucket.objects.filter(Prefix=l_folder).delete()


def merge_leg():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    client = boto3.client('s3')
    user_folders = s3_helper.list_folders(BUCKET_NAME, ROOT_FOLDER, client)
    for u_folder in user_folders:
        user_id = u_folder.split('/')[-2]
        print(f'user {user_id}')
        game_folders = s3_helper.list_folders(BUCKET_NAME, u_folder, client)
        for g_folder in game_folders:
            race_id = g_folder.split('/')[-2]
            print(f'race {race_id}')
            leg_merged = 1
            if race_id == '477':
                leg_merged = 4
            elif race_id == '480':
                leg_merged = 2
            legacy_folders = s3_helper.list_folders(
                BUCKET_NAME, g_folder, client)
            l_content = []
            for j, l_folder in enumerate(legacy_folders):
                print(f"Folder {j}/{len(legacy_folders)}")
                l_folder_name = l_folder.split('/')[-2]
                if l_folder_name == f"leg_{leg_merged}":
                    l_content = s3_helper.read_concat_csv_boat_status(
                        BUCKET_NAME, f"{l_folder}user_race_data.csv", s3)
            if l_content != []:
                csv_buffer = StringIO()
                key = f'{g_folder}leg_{leg_merged}/user_race_data.csv'
                content = s3_helper.read_concat_csv_boat_status(
                    BUCKET_NAME, key, s3)
                content_df = pd.DataFrame(content)
                content_df.columns = content_df.iloc[0]
                content_df.drop(content_df.index[0], inplace=True)

                l_content_df = pd.DataFrame(l_content)
                l_content_df.columns = l_content_df.iloc[0]
                l_content_df.drop(l_content_df.index[0], inplace=True)

                appended_df = content_df.append(l_content_df).drop_duplicates()
                appended_df.to_csv(csv_buffer, header=True, index=False)

                s3.Object(BUCKET_NAME, key).put(
                    Body=csv_buffer.getvalue())


def remove_unlegged():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    client = boto3.client('s3')
    user_folders = s3_helper.list_folders(BUCKET_NAME, ROOT_FOLDER, client)
    for u_folder in user_folders:
        user_id = u_folder.split('/')[-2]
        print(f'user {user_id}')
        game_folders = s3_helper.list_folders(BUCKET_NAME, u_folder, client)
        for g_folder in game_folders:
            race_id = g_folder.split('/')[-2]
            print(f'race {race_id}')
            key = f'{g_folder}user_race_data.csv'
            if s3_helper.key_exists(BUCKET_NAME, key, s3):
                print(f'Deleting {key}')
                bucket.objects.filter(Prefix=key).delete()


remove_unlegged()
'''
store_user_data()
remove_players_missing_data()
remove_unwanted_players()
merge_legacy_files()
remove_legacy_files()
merge_leg()
'''
