from app import application
from flask import render_template, flash, redirect, url_for
from flask import request 
from app.forms import chooseCountryForm
from app.serverlibrary import get_country, get_unique_countries, draw_plots
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64

@application.route('/')
@application.route('/index')
def index():
	return render_template('index.html', title='Home')

@application.route('/data', methods=['GET','POST'])
def data():
	form = chooseCountryForm()
	form.countries.choices=[country for country in get_unique_countries()]
	if form.validate_on_submit():
		country = get_country(form.countries.data)
		images = draw_plots(form.countries.data)
		return render_template('data.html', title='Data', tables=[country.to_html(justify="center",col_space=120)], form=form, images=images)

	return render_template('data.html', title='Data', tables=[], form=form, image = None)

