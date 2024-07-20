import pandas as pd
from IPython.core.display import display, HTML

def find_item(item_list, look_word):
    """
    To find look_word in the list provided
    parameters
    ----------
    item_list : a list of string items
    look_word : the word to be found in the list of items

    returns a list of items tha contains the look_word
    """
    return [x for x in item_list if look_word in x.lower()]

def text_to_html_table(text, row_sep='\n', head_sep='\t', 
    first_as_index=False, show=False):
    """
    To return html format of a table from text data.
    parameters
    ----------
    text : the text to parse with consistant row and header separators
    row_sep : the delimiter used to split the rows
    head_sep : the dilimeter used to split the columns
    first_as_index : to indicate if the first column should be indexed
    show : True means the html table will be just displayed
        False means the dataframe will be returned
    """
    data = text.split(row_sep)
    headers = data[0].split(head_sep)
    text_rows = [x.split(row_sep) for x in data[1:]]
    df = pd.DataFrame([x.split(head_sep) for x in text_rows for x in x], 
            columns=headers)    
    
    if first_as_index:
        df = df.set_index(headers[0])        
        if show:
            output = df.to_html().replace(row_sep, '')
            display(HTML(output))
        else:
            return df
    else:     
        if show:
            output = df.to_html(index=first_as_index).replace(row_sep, '')
            display(HTML(output))
        else:
            return df
