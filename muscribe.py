from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
@app.route('/<username>')
def main(username="Shruti"):
	return render_template("home.html", MIDI=None, au=None, output=None)

@app.route('/gmidi')
def gmidi():
	return render_template("home.html", MIDI=["a", "b"], au=None, output=None)

@app.route('/gmidi/combine')
def combine():
	return render_template("home.html", MIDI=["a", "b"], au="y", output=None)

@app.route('/predict')
def pr():
	return render_template("home.html", MIDI=["a", "b"], au="y", output=["a", "b"])

if __name__ == '__main__':
   app.run(debug = True)
