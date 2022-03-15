# Virtual Regatta: lambda functions repos

This folder holds lambda functions (not exhaustive, some other lambdas are present in the AWS account but not stored here) used for the Virtual Regatta project.

Folder structure:

- events: JSON files used for testing lambdas (extracted from AWS test events)
- race_data: JSON files holding information about ongoing races in Virtual Regatta
- src: the lambda functions + supporting code (e.g. s3_helper)
- test: the test files (*.py), that make use of the JSON event files
- zip: a folder only used to temporarly store lambda artifacts before they are uploaded

## Setup

- Preferably, use Pipenv. (Pipenv installation steps are extracted from: <https://pipenv.pypa.io/en/latest/#install-pipenv-today>)

```bash
pip install --user pipenv
```

```bash
pipenv install -d
```

The locking may take a while.  
See the troubleshooting section if you into issues here.  
The option -d installs the development packages in addition to the runtime ones: linter, formatter,...

```bash
pipenv shell
```

- Alternatively, use pip (not recommended)

```bash
pip install -r requirements.txt
```

Copy the environnement file from the sample:

```bash
cp .env_sample .env
```

Make sure you have the AWS CLI installed and configured.

Make sure to have zip installed:

```bash
sudo apt-get install zip
```

## Specific instructions per lambda function

### vr_parse_wnd

Trigger is a SNS topic (hosted in VR AWS account), ARN: arn:aws:sns:eu-west-1:380438412770:snsAddWinds

As VR is storing the WND files for ~1 month and then removes them, the test event might "expire", resulting in "file not found" or "access denied".
If so, look for: *\"key\": \"winds/live* in the JSON file, and replace the value with the current date or the day before: *\"key\": \"winds/live/20211004/00/063.wnd\"*

## vr_api_getboatstate

It's the only non-Python (JS) lambda (so far). Its purpose is to call the vr_api_websocket web service. It needs (at least) an user ID and a race ID.

How to find the race ID?

- Install the VR dashboard Chrome extension: <https://chrome.google.com/webstore/detail/vr-dashboard/amknkhejaogpekncjekiaolgldbejjan>
- Go to VRO (using Chrome): <https://www.virtualregatta.com/en/offshore-game/>
- Login, join a race
- The VR dashboard extension should pop up as a new tab, and you can see the race ID from the dropdown list at the top (typically a 3 figures number: 430, 502, etc.)

How to find the user ID?

- Use the VR dashboard extension as well, and navigate to the logs tab
- Your personal user ID will be present in most of the log messages
- If you don't see any logs, try changing your heading in the game (or any other action)
- To get the user ID of other players, you'll have to follow them in the game, and wait for a specific log message to be emitted (every 5 mins). It looks like the content of the file *data_vr_dashboard*
- You can use the script preprocess_data_vr_dashboard to export the data in a structure that can be used by the lambda function

## Deployment

Once you finsihed updating one or several lambda functions, first test your changes locally:

```bash
pytest
```

Optionnaly test one lambda with:

```bash
python test/vr_....py
```

You can update the test payload in the events folder.

If all is well:

```bash
chmod +x deploy.sh (only needed once)
./deploy.sh
```

Don't forget to update any other artifacts online (e.g. environnement variables)

## Maintenance

- If you update the packages list, make sure to update the requirements file accordingly. The best way to do so is to use pipenv for your local environnement:

```bash
pipenv install my_package
pipenv lock -r > requirements.txt
```

- Also update the third party licenses files (cf. <https://github.com/ftpsolutions/python-third-party-license-file-generator>):

```bash
rm THIRDPARTYLICENSES_summary
python -m third_party_license_file_generator -r requirements.txt -p $(which python) -g -c >> THIRDPARTYLICENSES_summary
```

(Remove the options -g -c if you want to get an error message for GPL/commercial licenses detected)

- Document! And update this Readme :)

## Troubleshooting

### Pipenv

- Pipenv not found by the system: you might need to update some Ubuntu files and/or restart your system: <https://superuser.com/questions/1432768/how-to-properly-install-pipenv-on-wsl-ubuntu-18-04>
- Runtime error when using pipenv: try specifying the Python version to use when creating the environment, for instance:

```bash
pipenv --three --python=`which python3`
```

(from <https://github.com/pypa/pipenv/issues/3363>)
