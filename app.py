from flask import Flask, render_template, redirect, url_for
import subprocess
import os
import signal

app = Flask(__name__)

# Store process globally
process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run')
def run():
    global process
    if process is None:
        process = subprocess.Popen(['python', 'Final.py'])
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global process
    if process is not None:
        process.terminate()
        process.wait()
        process = None
    return render_template('index.html', message="Code ran successfully and was stopped.")

if __name__ == '__main__':
    app.run(debug=True)
