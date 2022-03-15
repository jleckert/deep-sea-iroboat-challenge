# Remove previous zip files
find ./zip ! -name '.gitignore' -type f -exec rm -f {} +

# Parse WND files
cd ./src/
cp ./vr_parse_wnd.py ./lambda_function.py
zip ../zip/vr_parse_wnd.zip ./lambda_function.py
rm ./lambda_function.py
aws lambda update-function-code --function-name vr_parse_wnd --zip-file fileb://../zip/vr_parse_wnd.zip
aws lambda publish-version --function-name vr_parse_wnd

# Trigger isochrones computation
cp ./vr_trigger_isochrones_computation.py ./lambda_function.py
zip ../zip/vr_trigger_isochrones_computation.zip ./lambda_function.py ./s3_helper.py
rm ./lambda_function.py
cd ..
zip -ur ./zip/vr_trigger_isochrones_computation.zip ./race_data
aws lambda update-function-code --function-name vr_trigger_isochrones_computation --zip-file fileb://./zip/vr_trigger_isochrones_computation.zip
aws lambda publish-version --function-name vr_trigger_isochrones_computation

# Trigger isochrones update
cd ./src/
cp ./vr_trigger_isos_update.py ./lambda_function.py
zip ../zip/vr_trigger_isos_update.zip ./lambda_function.py ./s3_helper.py
rm ./lambda_function.py
cd ..
zip -ur ./zip/vr_trigger_isos_update.zip ./race_data
aws lambda update-function-code --function-name vr_trigger_isos_update --zip-file fileb://./zip/vr_trigger_isos_update.zip
aws lambda publish-version --function-name vr_trigger_isos_update

# Predict Heading
cd ./src/
cp ./vr_predict_heading.py ./lambda_function.py
zip ../zip/vr_predict_heading.zip ./lambda_function.py ./s3_helper.py ./geometry_helper.py
rm ./lambda_function.py
aws lambda update-function-code --function-name vr_predict_heading --zip-file fileb://../zip/vr_predict_heading.zip
aws lambda publish-version --function-name vr_predict_heading

# Process Script Message
cp ./process_script_message.py ./lambda_function.py
zip ../zip/process_script_message.zip ./lambda_function.py
rm ./lambda_function.py
aws lambda update-function-code --function-name process_script_message --zip-file fileb://../zip/process_script_message.zip
aws lambda publish-version --function-name process_script_message


concatenv_isos_sup="{"
input="../.env_isos_supervision"
echo "Environnement variables:"
while IFS= read -r line
do
  echo "$line"  
  concatenv_isos_sup+=$line
  concatenv_isos_sup+=","
done < "$input"
concatenv_isos_sup=${concatenv_isos_sup%,}
concatenv_isos_sup+="}"

# Isos supervision
cp ./vr_isos_supervision.py ./lambda_function.py
zip ../zip/vr_isos_supervision.zip ./lambda_function.py ./s3_helper.py
rm ./lambda_function.py
aws lambda update-function-configuration --function-name vr_isos_supervision --layers arn:aws:lambda:eu-west-2:770693421928:layer:Klayers-python38-numpy:12 arn:aws:lambda:eu-west-2:281633979087:layer:python-3-8-scikit-learn-0-23-1:1
aws lambda update-function-configuration --function-name vr_isos_supervision --timeout 60
aws lambda update-function-configuration --function-name vr_isos_supervision --environment Variables=$concatenv_isos_sup
aws lambda update-function-code --function-name vr_isos_supervision --zip-file fileb://../zip/vr_isos_supervision.zip
aws lambda publish-version --function-name vr_isos_supervision