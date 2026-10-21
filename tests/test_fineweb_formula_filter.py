"""Formula filtering must preserve ordinary financial/scientific prose."""
from scripts.campaigns.fineweb_formula_filter import classify


def test_financial_dates_currency_and_percentages_are_retained():
    text = 'In 1989 revenue rose 2.5% to $20 million. Shares cost $10 and $12 on Nov. 4.'
    assert classify(text)['decision'] == 'keep'


def test_medical_sample_sizes_and_p_values_are_retained():
    text = 'Patients received 5 mg/kg. In the study (n=154), p = 0.03 and OR = 2.78; 95% CI 1.18-6.56.'
    assert classify(text)['decision'] == 'keep'


def test_later_ratios_do_not_turn_a_sample_size_into_a_formula():
    text = 'The study (n=12) measured E. coli. The WHO standard was 0/100 ml.'
    assert classify(text)['decision'] == 'keep'


def test_parameter_estimates_and_intervals_are_retained():
    text = 'Bayesian length-weight estimates: a=0.00363 (0.00140 - 0.00941), b=3.11 (2.88 - 3.34).'
    assert classify(text)['decision'] == 'keep'


def test_observed_ratio_values_and_fractional_sample_counts_are_retained():
    for text in ['Ka / Ks = 1 indicates neutrality.', 'The report states L / D=4.', 'Patients in the subgroup (n = 19/25) were observed.', 'Score = 16/20.']:
        assert classify(text)['decision'] == 'keep', text


def test_currency_conversion_descriptions_are_retained():
    for text in ['$ 142.85 Pesos = $ 10 US.', 'Zloty = 1/5 of the quoted currency unit.']:
        assert classify(text)['decision'] == 'keep', text


def test_scientific_unit_conversion_is_retained():
    text = 'The lifetime is a few picoseconds (1 ps = 10 - 12 s).'
    assert classify(text)['decision'] == 'keep'


def test_paired_currency_and_php_variables_are_not_latex():
    for text in ['$ 10 (USD) membership fee and a = $ 50 sign-up bonus.', 'The values $ _ GET, $ _ POST and $ _ REQUEST are array variables.']:
        assert classify(text)['decision'] == 'keep'


def test_quoted_metadata_identifier_is_retained():
    assert classify("link_id=’484e2618 - 991a-11e7 - b0c4 - 19ac8375cbb8′]")['decision'] == 'keep'


def test_ptb_markers_and_table_statistics_are_retained():
    for text in ['-LRB-n = 370-RRB-|', 'N=11||Interventions targeted at children aged 10 - 15.', 'P=.01). Mortality fell from 6.3 to 2.5 per 100,000.']:
        assert classify(text)['decision'] == 'keep'


def test_unit_symbols_dates_http_headers_and_resolution_are_retained():
    for text in ['Mean = 15.4 μm; Cres = 1 μF.', 'date=9 April 2013 |retrieved 2014 - 08 - 16', 'Cache-control: max - age=31536000', '4 K (UHD) = 3840×2160 (2160p) resolution.']:
        assert classify(text)['decision'] == 'keep', text


def test_shell_variables_currency_codes_and_bbcode_are_retained():
    for text in ['$errlvl == 3 goto redial; if $errlvl != 0 goto try-again', '$1 = us$ 1.07', '$ [url = http://example.org/x]Link-0.86$']:
        assert classify(text)['decision'] == 'keep', text


def test_chemical_double_bond_notation_is_retained():
    assert classify('The fatty acid chain is -CH = CH-(CH2)7-COOH.')['decision'] == 'keep'


def test_an_unlabelled_numeric_range_is_not_auto_dropped():
    assert classify('Participant B = 38 - 43 years old.')['decision'] != 'drop'


def test_chemical_names_and_clinical_units_are_retained():
    assert classify('CO2 and H2O were measured. Blood pressure was 120/80 mmHg and temperature 37°C.')['decision'] == 'keep'


def test_math_topic_without_formulas_is_retained():
    text = 'Math teachers help students understand algebra and geometry. Classroom learning benefits from patient teaching.'
    assert classify(text)['decision'] == 'keep'


def test_url_parameters_and_textual_definitions_are_retained():
    text = 'Read https://example.org/article?id=23&r=1+2. The Latin word animal = a living being.'
    assert classify(text)['decision'] == 'keep'


def test_latex_and_mathml_are_excluded():
    for text in [r'Compute \frac{x+1}{y}.', r'\begin{align} x+y &= 2 \end{align}', '<math><mi>x</mi></math>', r'$1+1=2$']:
        assert classify(text)['decision'] == 'drop', text


def test_algebra_and_worked_numeric_equations_are_excluded():
    for text in ['The equation x + y = 3 is linear.', 'Einstein wrote E = mc^2.', 'RTY = 40/50 * 34/46 * 37/46']:
        assert classify(text)['decision'] == 'drop', text


def test_greek_physics_formula_is_excluded():
    assert classify('Heat transfer follows the formula Q = mcΔT, where c is heat capacity.')['decision'] == 'drop'
    assert classify('The equation q = √60/1000 gives the desired value.')['decision'] == 'drop'


def test_code_is_not_automatically_excluded_by_assignment():
    assert classify('function increment(x) { const y = x + 1; return y; }')['decision'] == 'review'
    assert classify('def increment-LRB-y-RRB-: x = 0; x = x + y')['decision'] != 'drop'


def test_uncertain_symbolic_assignment_is_not_confirmed():
    assert classify('The equation uses a coefficient and a variable. Let x = a.')['decision'] == 'review'
