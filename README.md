# New_optimCriteries
Фалй json с http://jmcauley.ucsd.edu/data/amazon/ обрабатываются программой на https://github.com/ValeriaSaf/Data
файл main содержит функции: 
Get_applicants() - обработка сырых текстовых отзывов и извлечение кандидатов, для обработки с помощью теггера NLTK вконце функции вызвать
                   get_word_applicant(), для обработки с помощью Stanford parser последовательно вызвать функции: get_word_application_Stanford(),
                   clear_tag_Stanford() и list_clear_feature()
Get_clear_features() - обрабатывает кандидиатов, выбирает значимые критерии. Для работы с кандидатами теггера NLTK функция должна принять файл "Features.txt"
                       Для работы с кандидатами теггера Stanford parser функция должна принять файл "Features_Stanford.txt"
Get_vector_features() 
Classifiers()

