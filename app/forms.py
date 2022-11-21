from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectMultipleField, IntegerField, HiddenField
from wtforms.validators import DataRequired, ValidationError, EqualTo

class chooseCountryForm(FlaskForm):
	countries = SelectMultipleField('Select Countries', validators=[DataRequired()])
	submit = SubmitField('Submit')