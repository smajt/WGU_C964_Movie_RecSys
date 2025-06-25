import numpy as np
from nicegui import ui, events, run
from io import StringIO
import pandas as pd
import asyncio
from cf_recommender_system import RecSys

if __name__ in {'__main__', '__mp_main__'}:
    global_movies_df = None
    global_ratings_df = None
    global_rec_sys = RecSys()
    global_eta = None
    top_n_csv = None

    def get_eta():
        global global_eta
        global_eta -= 1

        return global_eta

    async def ticker():
        while True:
            await asyncio.sleep(1)
            print("Working...")


    def csv_handler_movies(event: events.UploadEventArguments):
        with StringIO(event.content.read().decode()) as f:
            global global_movies_df
            global_movies_df = pd.read_csv(f)

    def csv_handler_ratings(event: events.UploadEventArguments):
        with StringIO(event.content.read().decode()) as f:
            global global_ratings_df
            global_ratings_df = pd.read_csv(f)

    async def handle_train_model_click():
        grid_col.clear()
        loop = asyncio.get_event_loop()
        asyncio.create_task(ticker())
        spin.visible=True
        label.visible=True
        train_button.disable()
        global global_eta
        global_eta = 16
        eta_timer = ui.timer(60.0, lambda: label.set_text('(ETA ' + str(get_eta()) + ' Minutes)'))
        global global_rec_sys
        global_rec_sys.set_movie_rating_df(global_movies_df, global_ratings_df)
        await loop.run_in_executor(None, global_rec_sys.make_datasets)
        insert_viz_1(global_rec_sys.rating_list, global_rec_sys.genre_list)
        insert_viz_2(global_rec_sys.all_ratings)
        await loop.run_in_executor(None,global_rec_sys.train_eval_algos)
        insert_viz_3(global_rec_sys.mark_scores, global_rec_sys.names)
        await loop.run_in_executor(None,global_rec_sys.train_rec_sys_algo)
        global top_n_csv
        grid_col.clear()
        top_n_csv = global_rec_sys.top_n_df.to_csv(index=False)
        insert_result_grid(global_rec_sys.top_n_df)
        train_button.enable()
        spin.visible=False
        label.visible=False
        eta_timer.deactivate()

    def insert_result_grid(top_n_df):
        global top_n_csv
        with grid_col.classes('w-full no-wrap'):
            ui.aggrid.from_pandas(top_n_df).classes('w-full')
            ui.button('Save Output', on_click=lambda: ui.download.content(top_n_csv, 'out.csv'))


    def insert_viz_1(avg_rating_list, genre_list):
        viz_1_label.visible=False
        viz_1_row.clear()
        with viz_1_row:
            with ui.matplotlib(figsize=(7,5),tight_layout=True).figure as fig:
                y_pos = np.arange(len(genre_list))
                fig.subplots_adjust(left=.1)
                ax = fig.gca()
                ax.set_title('Average Rating by Genre')
                ax.set_yticks(y_pos, labels=genre_list)
                ax.invert_yaxis()
                ax.set_ylabel('Genre')
                ax.set_xlabel('Rating')
                ax.set_xlim(right=5.0)
                ax.barh(y_pos, avg_rating_list)

    def insert_viz_2(all_ratings):
        viz_2_label.visible=False
        viz_2_row.clear()
        with viz_2_row:
            with ui.matplotlib().figure as fig:
                hist, edges = np.histogram(all_ratings, bins=10, density=True, range=(0.5, 5))
                edges = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
                width = 0.5
                ax = fig.gca()
                ax.set(xticks=np.arange(0.5, 5.5, 0.5),
                       xlim=[0, 5.5],
                       xlabel='Rating',
                       ylabel='Density',
                       title='Distribution of Reviews by Rating')
                ax.bar(edges, hist, width=width, align='center', edgecolor='white')

    def insert_viz_3(mark_scores, names):
        viz_3_label.visible=False
        viz_3_row.clear()
        with viz_3_row:
            with ui.matplotlib().figure as fig:
                index = np.arange(1,11)
                ax = fig.gca()
                ax.set(xticks=index,
                       xlabel='K',
                       ylabel='MAR@K',
                       title='Mean Average Recall at K (MAR@K)')
                ax.plot(index, mark_scores[0], label=names[0], linestyle='-', linewidth=5)
                ax.legend()


    with ui.header().classes(replace='row items-center') as header:
        with ui.tabs() as tabs:
            ui.tab('The Recommender System')
            ui.tab('Visualization 1')
            ui.tab('Visualization 2')
            ui.tab('Visualization 3')
    with ui.tab_panels(tabs, value='The Recommender System').classes('w-full'):
        with ui.tab_panel('The Recommender System'):
            with ui.row():
                with ui.column():
                    ui.upload(label='Upload Ratings file (Only supports .csv files)',  auto_upload=True, on_upload=csv_handler_ratings).classes('max-w-full')
                with ui.column():
                    ui.upload(label='Upload Movies file (Only supports .csv files)', auto_upload=True, on_upload=csv_handler_movies).classes('max-w-full')
            with ui.row().classes('place-content-center'):
                train_button = ui.button('Train Model', on_click=lambda: handle_train_model_click())
                spin = ui.spinner('gears', size='lg')
                spin.visible=False
                label = ui.label()
            grid_col = ui.column()

        with ui.tab_panel('Visualization 1'):
            with ui.row():
                viz_1_label = ui.label('The model is still training. Visualizations will populate when ready.')
                viz_1_row = ui.row()
        with ui.tab_panel('Visualization 2'):
            with ui.row():
                viz_2_label = ui.label('The model is still training. Visualizations will populate when ready.')
                viz_2_row = ui.row()
        with ui.tab_panel('Visualization 3'):
            viz_3_label = ui.label('The model is still training. Visualizations will populate when ready.')
            viz_3_row = ui.row()

    ui.run()