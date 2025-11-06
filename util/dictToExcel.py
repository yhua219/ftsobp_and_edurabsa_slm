import pandas as pd

def df_dictToExcel(excelfilename, sheetname_df_dict, freezeheader=True, headerfilter=True, locksheet=False,
        editable_range=None):
    protect_options = {
        'objects': True,
        'scenarios': True,
        'format_cells': True,
        'format_columns': True,
        'format_rows': True,
        'insert_columns': True,
        'insert_rows': True,
        'insert_hyperlinks': True,
        'delete_columns': False,
        'delete_rows': False,
        'select_locked_cells': True,
        'sort': True,
        'autofilter': True,
        'pivot_tables': True,
        'select_unlocked_cells': True,
    }

    with pd.ExcelWriter(excelfilename) as writer:
        for sheetname, df in sheetname_df_dict.items():

            df.to_excel(writer, sheet_name=sheetname, index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheetname]

            # ======================= format header cell ============================

            header_format = workbook.add_format(
                {'bold': True, 'valign': 'top', 'align': 'left', 'bg_color': '#D9D9D9'})  # '#F9DA04'})
            specialcell_format = workbook.add_format(
                {'bold': True, 'valign': 'top', 'align': 'left', 'bg_color': '#92D050'})

            headerlist = list(df.columns.values)
            for i in range(len(headerlist)):
                header = str(headerlist[i])
                if 'MANUAL' in header:
                    worksheet.write(0, i, header, specialcell_format)
                else:
                    worksheet.write(0, i, header, header_format)

                    # ======================= format embedded url cell ========================

            if 'Filelink' in set(df.columns.values):  # format FileLink column to url style

                urlformat = urlformat = workbook.get_default_url_format()  # or: workbook.add_format({'font_color':'blue', 'underline':'true'})

                col_pos = chr(df.columns.get_loc('Filelink') + 65)
                worksheet.set_column(f'{col_pos}:{col_pos}', None, urlformat)

                # ======================== other formatting ===============================

            if freezeheader:
                worksheet.freeze_panes(1, 0)

            if headerfilter:
                worksheet.autofilter(0, 0, df.shape[0],
                                     df.shape[1] - 1)  # args:  first_row, first_col, last_row, last_col

            if locksheet:
                worksheet.protect('', protect_options)  # first arg is password str

                if editable_range != [] and editable_range != None:
                    for eachcol in editable_range:
                        worksheet.unprotect_range(eachcol)

            # ========================================================================

            print(f"\033[33m{sheetname}\033[0m written to \033[33m{excelfilename}.\033[0m")
    print("\n")

# =====================================================================================================================


# adict = dict()

# df1 = pd.DataFrame({'a':[1,2,3]})
# df = pd.DataFrame({'MANUAL_NOTE':[1,2,3], 'b':[4,5,6], 'c':[7,8,9], 'd':[0, 0, 0]})
# img = 'Table5_graph.png'

# adict['1'] = {'type':'df', 'item': df1, 'freezeheader':False, 'headerfilter':False, 'locksheet':True, 'editable_range': None}
# adict['2'] = {'type':'df', 'item': df}#, 'freezeheader':True, 'headerfilter':True, 'locksheet':True, 'editable_range': ['A:A', 'B:B']}
# adict['3'] =  {'type':'both', 'item': {'dfs':[df, df1, df1], 'images':[img], 'itemspacing':(3, 0), 'typespacing':(0, 3)}, 'locksheet':True, 'editable_range': ['D2', 'G:G']}
# dictToExcel('df.xlsx', adict)

# df_dictToExcel('df2.xlsx', {'df1': df1, 'df':df}, freezeheader=True, headerfilter=True, locksheet=True, editable_range=['A:A', 'B3'])
