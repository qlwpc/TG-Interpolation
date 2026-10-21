from scripts.campaigns.fineweb_collection_filter import classify


def test_known_function_formula_gap_is_closed():
    for text in ['u(x ) = ln(x )', 'f(x) = sin(x)', 'g(t) = 3.5']:
        assert not classify(text)['keep']


def test_boilerplate_is_excluded_without_cookie_topic_filter():
    for text in ['Accept all cookies', 'This website uses cookies.', 'Leave a Reply',
                 'Cookie preferences', 'Please enable JavaScript']:
        assert 'web_boilerplate' in classify(text)['reasons']
    assert classify('Bake the chocolate cookies for ten minutes.')['keep']


def test_finance_medical_and_math_prose_are_kept():
    for text in ['Shares rose 2.5% to $20 million in 1989.',
                 'Patients received 5 mg/kg; n = 154, p = 0.03 and OR = 2.78.',
                 'Algebra teachers use patient instruction to help students.',
                 'The browser can store HTTP cookies for authentication.']:
        assert classify(text)['keep'], text


def test_damage_and_dominant_repetition_are_excluded():
    assert not classify('WÃ¼ster studied the subject.')['keep']
    assert not classify('This is uncorrected OCR text.')['keep']
    assert not classify(('This repeated advertisement is long enough to pass the long line threshold. ' * 2 + '\n') * 4)['keep']


def test_table_or_code_is_not_excluded_by_topic_alone():
    assert classify('Species | Count | Habitat\nSparrow | 30 | Woodland\nCrow | 8 | Urban')['keep']


def test_medical_problem_not_corrected_is_not_an_extraction_warning():
    assert classify('Underweight signifies poor growth which, if not corrected, harms health.')['keep']
