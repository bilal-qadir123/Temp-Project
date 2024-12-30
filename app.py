from flask import Flask, render_template, request, redirect, url_for, session, make_response
from flask import request, jsonify
from model import preprocess_user_input, predict_user_input
import re

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = 'your_secret_key_here'

# Route for handling login form (Sign-in)
@app.route('/', methods=['GET', 'POST'])
def login():
    if 'email' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        found = False

        if email == 'admin@123' and password == 'admin':
            return redirect(url_for('admin'))

        with open('accounts.txt', 'r') as file:
            accounts = file.readlines()

            for account in accounts:
                account = account.strip()
                if not account:
                    continue
                try:
                    stored_email, stored_password = account.split(',', 2)[:2]  # Only consider first two parts
                except ValueError:
                    continue
                if stored_email == email and stored_password == password:
                    session['email'] = email
                    return redirect(url_for('index'))

        return render_template('login.html')

    return render_template('login.html')

# Route for handling sign-up form submission
@app.route('/signup', methods=['POST'])
def signup():
    email = request.form['email']
    password = request.form['password']

    if email == 'admin@123':
        return render_template('login.html')

    # Check if the email already exists in the accounts.txt file
    with open('accounts.txt', 'r') as file:
        accounts = file.readlines()
        for account in accounts:
            account = account.strip()
            if not account:
                continue
            if account.split(',')[0] == email:  # Check only the email part
                return render_template('login.html')  # Email already exists, do nothing

    # If email doesn't exist, create a new account
    with open('accounts.txt', 'a') as file:
        file.write(f"{email},{password}\n")  # Add email and password with a newline

    session['email'] = email
    return redirect(url_for('index'))
# Route for handling sign-out
@app.route('/signout')
def signout():
    session.pop('email', None)  # Clear the email from the session

    # Create the response and set no-cache headers
    response = make_response(redirect(url_for('login')))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Route for handling index page
@app.route('/index', methods=['GET', 'POST'])
def index():
    email = session.get('email')
    if not email:
        return redirect(url_for('login'))

    guide_content = ""  # Default value if no guide is found
    guide_present = False  # Default to False if guide isn't found

    # Open accounts.txt to check if guide content exists for the user
    with open('accounts.txt', 'r') as file:
        accounts = file.readlines()
        for account in accounts:
            account = account.strip()
            if not account:
                continue
            parts = account.split(',', 2)  # Split into email, password, and possibly guide
            if len(parts) >= 3 and parts[0] == email:  # Check if email matches
                guide_present = True  # If guide is found
                guide_content = parts[2]  # The third part is the guide text
                break

    if request.method == 'POST':
        user_input = preprocess_user_input(request.form)
        prediction = predict_user_input(user_input)

        if prediction == 1:
            prediction_text = "Predicted Stress Type: Eustress (Moderate level)"
        elif prediction == 2:
            prediction_text = "Predicted Stress Type: No Stress (0 or very low)"
        else:
            prediction_text = "Predicted Stress Type: Distress (High)" 

        # Collect user input details
        user_details = {key: request.form[key] for key in request.form}

        # Redirect to the result page with prediction and user details
        return redirect(
            url_for('result', prediction=prediction_text, **user_details)
        )
    
    print(f"Guide present: {guide_present}")
    print(f"Guide content: {guide_content}")
    return render_template('index.html', email=email, guide_present=guide_present, guide_content=guide_content)


# Route for displaying prediction result
@app.route('/result')
def result():
    email = session.get('email')
    prediction = request.args.get('prediction', 'No Prediction Available')

    # Extract user details from request args
    user_details = {key: request.args.get(key) for key in request.args if key != 'prediction'}
    user_details_str = ", ".join([f"{key}={value}" for key, value in user_details.items()])

    # Read existing data from predictions.txt
    updated = False
    new_lines = []
    with open('predictions.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            stored_email = line.split(',')[0].strip()
            if stored_email == email:
                # Overwrite the line if email matches
                new_lines.append(f"{email}, {prediction}, {user_details_str}\n")
                updated = True
            else:
                new_lines.append(line)

    # If no matching email, append a new line
    if not updated:
        new_lines.append(f"{email}, {prediction}, {user_details_str}\n")

    # Write back the updated data to predictions.txt
    with open('predictions.txt', 'w') as file:
        file.writelines(new_lines)

    # Print details to the terminal for debugging
    print(f"Stored Data: Email={email}, Prediction={prediction}, Details={user_details_str}")

    return render_template('result.html', prediction=prediction, email=email, user_details=user_details)



@app.route('/admin', methods=['GET'])
def admin():
    email = "admin@123"  # Admin's email (hardcoded for this example)

    # Prepare lists to store data
    email_prediction_details = []
    email_set = set()  # To track emails that we've already processed

    with open('predictions.txt', 'r') as file:
        for line in file:
            line = line.strip()

            # Split the line into email, prediction, and details
            parts = line.split(', ', 2)  # Split into exactly 3 parts: email, prediction, and details

            # Check if the line has all three parts (email, prediction, details)
            if len(parts) == 3:
                email_data, prediction, details = parts
                prediction = prediction[22:]  # Remove the first 22 characters of the prediction
            else:
                print(f"Skipping malformed line: {line}")  # Debugging: log malformed lines
                continue

            # Check if this email has been processed before
            if email_data not in email_set:
                email_set.add(email_data)
                email_prediction_details.append((email_data, prediction, details))
            else:
                print(f"Duplicate email found and skipped: {email_data}")  # Debugging: log duplicate emails
                continue

    # Debugging: Verify data read and print
    print("\nProcessed Email Prediction Details:")
    for detail in email_prediction_details:
        print(f"Email: {detail[0]}\nPrediction: {detail[1]}\nDetails: {detail[2]}\n{'-'*50}")

    # Calculate the counts
    total_students = len(email_prediction_details)
    eustress_count = sum(1 for _, prediction, _ in email_prediction_details if "Eustress" in prediction)
    no_stress_count = sum(1 for _, prediction, _ in email_prediction_details if "No Stress" in prediction)
    distress_count = sum(1 for _, prediction, _ in email_prediction_details if "Distress" in prediction)

    # Render the admin page with the correct details
    return render_template(
        'admin.html',
        email_prediction_details=email_prediction_details,
        account=email,
        total_students=total_students,
        eustress_count=eustress_count,
        no_stress_count=no_stress_count,
        distress_count=distress_count
    )

@app.route('/mark_clicked', methods=['POST'])
def mark_clicked():
    data = request.get_json()
    email = data.get('email')
    guide_text = data.get('guide')  # Retrieve the custom guide text from the request

    if not email or not guide_text:
        return jsonify({"error": "Email and guide text are required"}), 400

    # Read and update accounts.txt
    updated_lines = []
    with open('accounts.txt', 'r') as file:
        lines = file.readlines()
        found = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 2 and parts[0] == email:
                updated_line = f"{line},{guide_text}"  # Add custom guide text instead of "Clicked"
                updated_lines.append(updated_line)
                found = True
            else:
                updated_lines.append(line)

        # If email was not found, return an error response
        if not found:
            return jsonify({"error": f"Email {email} not found"}), 404

    # Write back the updated content to accounts.txt
    with open('accounts.txt', 'w') as file:
        file.write('\n'.join(updated_lines) + '\n')

    return jsonify({"message": "Email marked with custom guide text"}), 200 

@app.route('/delete_guide', methods=['POST'])
@app.route('/delete_guide', methods=['POST'])
def delete_guide():
    email = session.get('email')
    if not email:
        return redirect(url_for('login'))
    
    # Read the accounts.txt file and update the guide associated with the email
    with open('accounts.txt', 'r') as file:
        accounts = file.readlines()
    
    with open('accounts.txt', 'w') as file:
        for account in accounts:
            parts = account.strip().split(',', 2)
            if len(parts) >= 3 and parts[0] == email:
                # If guide is present, remove it by not including it in the file
                parts[2] = ''  # Remove the guide part (everything after the second comma)
                account = ','.join(parts)  # Rebuild the line without the guide
            file.write(account + '\n')  # Write each line back to the file
    
    return jsonify({'status': 'success'})  # Send success response to the frontend



if __name__ == '__main__':
    app.run(debug=True)
