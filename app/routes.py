from app import application
from flask import render_template, flash, redirect, url_for
from flask import request 
from app.forms import chooseCountryForm
from app.serverlibrary import get_country, get_unique_countries, draw_plots, draw_plots_for_model, r2_both, get_mean_std, get_df_features_train

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

@application.route('/model', methods=['GET','POST'])
def model():
	form = chooseCountryForm()
	form.countries.choices=[country for country in get_unique_countries()]
	if form.validate_on_submit():
		images_1 = draw_plots_for_model(form.countries.data)
		mean, std = get_mean_std(get_df_features_train())
		images_2 = draw_plots_for_model(form.countries.data, mean, std)
		return render_template('model.html', title='Model', r2_1=r2_both(form.countries.data), r2_2=r2_both(form.countries.data,mean, std),form=form, images_1=images_1, images_2=images_2)

	return render_template('model.html', title='Model', r2_1=None, r2_2=None, form=form, images_1 = None, images_2 = None)

