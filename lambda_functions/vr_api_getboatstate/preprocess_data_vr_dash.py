import json

with open('data_vr_dashboard.json') as json_file:
    data = json.load(json_file)

list_user_ids=[]
for user in data['res']:
    list_user_ids.append(user['userId'])

set_user_ids = set(list_user_ids)

out_data = {"users":[]}
for user_id in set_user_ids:
    out_data["users"].append({
            "id": user_id,
            "races": [
                502
            ]
        })

with open('data.json', 'w') as outfile:
    json.dump(out_data, outfile)