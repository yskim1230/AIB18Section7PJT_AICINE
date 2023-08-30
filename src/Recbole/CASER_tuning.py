from recbole.trainer.hyper_tuning import HyperTuning
from recbole.quick_start import objective_function

################################################################################################################################
# 하이퍼 파라미터 튜닝 값 설정하는법
# params_dict = {
        #     'choice': { # 주어진 목록에서 값을 선택
        #         'param1': [value1, value2, ...],
        #         'param2': [value1, value2, ...],
        #         ...
        #     },
        #     'uniform': { # 주어진 범위에서 연속적인 랜덤 값
        #         'param1': [low, high],
        #         'param2': [low, high],
        #         ...
        #     },
        #     'quniform': { # 주어진 범위에서 일정 간격으로 값
        #         'param1': [low, high, q],
        #         'param2': [low, high, q],
        #         ...
        #     },
        #     'loguniform': { # 주어진 범위에서 연속적인 값을 로그 스케일로 변환.
        #         'param1': [low, high],
        #         'param2': [low, high],
        #         ...
        #     },
        # }
################################################################################################################################
def tuning():
        hp = HyperTuning(objective_function, 
                params_dict={
                        'uniform' : {
                                'learning_rate': [0.0005,0.003],
                                'reg_weight': [1e-5, 1e-3]
                        },
                        'choice': {
                                'embedding_size': [32, 64, 128],
                                'n_v': [2, 4, 8],
                                'n_h': [2, 4, 8],
                        }
                },
                fixed_config_file_list=[
                './CASER_config.yaml'
                ], 
                display_file='tuning_log_CASER.html',
                algo='exhaustive', 
                max_evals=30, 
                early_stop=10
                )

        # run
        hp.run()

        # export result to the file
        hp.export_result(output_file='hyper_example_CASER.result')

        # print best parameters
        print('best params: ', hp.best_params)

        # print best result
        print('best result: ')
        print(hp.params2result[hp.params2str(hp.best_params)])