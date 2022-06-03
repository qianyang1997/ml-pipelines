import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype
import plotly.express as px
import plotly.graph_objects as go


def beautify_table(df: pd.DataFrame, title: str, subtitle: str=''):
    """Beautify table.

    Args:
        df (pd.DataFrame): input table
        title (str): table title
        subtitle (str, optional): table subtitle. Defaults to ''.

    Returns:
        json: plotly table in json format
    """
    
    # ensure not to change the original df
    df1 = df.copy()
    
    for c in df1.columns:
        # non-numeric types
        if not is_numeric_dtype(df1[c]):
            df1[c] = df1[c].astype(str)
            df1[c] = df1[c].str.replace(r'\bnan\b', '', regex=True)
        # numeric types
        else:
            # percentage
            if '%' in c:
                df1[c] = df1[c].map('{:.0%}'.format)
                df1[c] = df1[c]
            elif '$' in c:
                if df1[c].abs().mean() > 10 ** 12:
                    df1[c] = df1[c] / (10 ** 12)
                    df1[c] = df1[c].map('${:,.2f}T'.format)
                elif df1[c].abs().mean() > 10 ** 9:
                    df1[c] = df1[c] / (10 ** 9)
                    df1[c] = df1[c].map('${:,.2f}B'.format)
                elif df1[c].abs().mean() > 10 ** 6:
                    df1[c] = df1[c] / (10 ** 6)
                    df1[c] = df1[c].map('${:,.2f}M'.format)
                else:
                    df1[c] = df1[c].map("${:,.0f}".format)
                df1[c] = df1[c].str.replace('-(.*)', r'(\1)', regex=True)
            # integer
            elif is_integer_dtype(df1[c]):
                df1[c] = df1[c].map("{:,.0f}".format)
            # decimal
            else:
                df1[c] = df1[c].map("{:,.2f}".format)
            # deal with missing values
            df1[c] = df1[c].str.replace('.*nan.*', '', regex=True)

    # visualize in plotly
    title = f'<b>{title}</b>'
    if subtitle:
        title += f'<br><sup>{subtitle}</sup>'
    plotly_table = go.Figure(
        layout_title_text=title,
        data=[go.Table(
        header=dict(values=list(df1.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=df1.transpose().values.tolist(),
                fill_color='lavender',
                align='left'))
    ])
    plotly_table.update_layout(height=400)
    
    return plotly_table.to_json()
        