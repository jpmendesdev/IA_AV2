import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Importa todas as classes e funções necessárias do arquivo de biblioteca
from CirilloLearning import (
    ensure_2d_array, add_bias, train_test_split, normalize_standard,
    confusion_matrix_custom, accuracy_from_confusion, precision_sensitivity_specificity_f1,
    Perceptron, Adaline, MLP, RBFNetwork,
    load_spiral_csv, load_recfac, one_hot, one_hot_standard,
    run_binary_classification_mc,
    plot_scatter, plot_confusion_matrix, plot_learning_curve, summary_statistics
)

#<<ETAPA 1>>
def format_summary_table(summary_table, metric_name):
    header = f"\n--- Tabela de Resultados para a Métrica: {metric_name.upper()} ---\n"
    columns = ["Modelos", "Média", "Desvio-Padrão", "Maior Valor", "Menor Valor"]
    col_widths = [len(c) for c in columns]
    model_names = list(summary_table.keys())
    
    if model_names:
        col_widths[0] = max(col_widths[0], max(len(name) for name in model_names))
        for name, summary in summary_table.items():
            col_widths[1] = max(col_widths[1], len(f"{summary['mean']:.4f}"))
            col_widths[2] = max(col_widths[2], len(f"{summary['std']:.4f}"))
            col_widths[3] = max(col_widths[3], len(f"{summary['max']:.4f}"))
            col_widths[4] = max(col_widths[4], len(f"{summary['min']:.4f}"))

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    table_str = header
    row_str = "|"
    for i, col in enumerate(columns):
        row_str += f" {col:<{col_widths[i]}} |"
    table_str += separator + "\n" + row_str + "\n" + separator
    
    for name in model_names:
        summary = summary_table[name]
        row_str = "|"
        row_str += f" {name:<{col_widths[0]}} |"
        row_str += f" {summary['mean']:>{col_widths[1]}.4f} |"
        row_str += f" {summary['std']:>{col_widths[2]}.4f} |"
        row_str += f" {summary['max']:>{col_widths[3]}.4f} |"
        row_str += f" {summary['min']:>{col_widths[4]}.4f} |"
        table_str += "\n" + row_str

    table_str += "\n" + separator + "\n"
    print(table_str)

def plot_box_violin_numpy(data_dict, metric_name, title):
    data_list = [np.array(v) for v in data_dict.values()]
    labels = list(data_dict.keys())

    plt.figure(figsize=(12, 5))

    # Boxplot
    plt.subplot(1, 2, 1)
    plt.boxplot(data_list, labels=labels)
    plt.title(f'Boxplot de {metric_name} - {title}')
    plt.ylabel(metric_name)
    plt.xticks(rotation=15, ha='right')

    # Violin Plot
    plt.subplot(1, 2, 2)
    plt.violinplot(data_list, showmeans=True)
    plt.xticks(np.arange(1, len(data_list) + 1), labels, rotation=15, ha='right')
    plt.title(f'Violin Plot de {metric_name} - {title}')
    plt.ylabel(metric_name)

    plt.tight_layout()
    plt.show()

def run_etapa1_classification_spiral():
    """Executa a Etapa 1: Classificação Binária para Problemas Bidimensionais """
    print("\n" + "="*70)
    print("ETAPA 1: Classificação Binária")
    print("="*70)

    try:
        X, y = load_spiral_csv('spiral_d.csv')
        y_labs = np.unique(y)
        print(f"Dados carregados: N={X.shape[0]}, p={X.shape[1]}, Classes={y_labs}")
        
        plot_scatter(X, y, title='Visualização Inicial dos Dados Spiral D')
        plt.show()

        rng_seed = 42
        R_val = 5
        
        hp_perc = {'lr': 0.01, 'epochs': 1000, 'bipolar': True, 'random_state': rng_seed}
        hp_adaline = {'lr': 0.001, 'epochs': 1000, 'bipolar': True, 'random_state': rng_seed}
        hp_mlp_base = {'hidden_sizes': [10], 'activation': 'tanh', 'lr': 0.01, 'epochs': 500, 'random_state': rng_seed}
        hp_rbf_base = {'n_centers': 10, 'random_state': rng_seed}

        print(f"\n5. Executando Validação Monte Carlo (R={R_val}) com 80/20...")
        
        all_results = {}
        model_configs = {
            'Perceptron Simples': {'name': 'perceptron', **hp_perc},
            'ADALINE': {'name': 'adaline', **hp_adaline},
            'MLP Multicamadas': {'name': 'mlp', **hp_mlp_base},
            'Rede RBF': {'name': 'rbf', **hp_rbf_base}
        }
        
        for name, config in model_configs.items():
            print(f"  -> Rodando {name}...")
            
            # CORREÇÃO: Separar 'name' de 'kwargs' antes de chamar run_binary_classification_mc
            model_name = config['name']
            model_kwargs = {k: v for k, v in config.items() if k != 'name'}
            
            results = run_binary_classification_mc(X, y, 
                                                   model_name=model_name, 
                                                   R=R_val, 
                                                   test_size=0.2, 
                                                   rng_seed=rng_seed, 
                                                   **model_kwargs)
            all_results[name] = results
            print(f"  -> {name} concluído.")

        print("\n7. Calculando estatísticas resumo e gerando tabela/gráficos...")
        metrics_to_report = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1']
        
        for metric in metrics_to_report:
            metric_data = {}; summary_table = {}
            for name, results in all_results.items():
                values = [res['metrics'].get(metric, 0.0) for res in results if 'metrics' in res]
                if not values: continue
                metric_data[name] = values
                summary_table[name] = summary_statistics(values)

            format_summary_table(summary_table, metric)
            plot_box_violin_numpy(metric_data, metric, 'Etapa 1')

    except FileNotFoundError as e:
        print(f"ERRO: {e}. Por favor, verifique se 'spiral_d.csv' está na pasta correta.")
    except Exception as e:
        print(f"ERRO durante a Etapa 1: {e}")
        # import traceback; traceback.print_exc()

def run_etapa2_multiclass_recfac():
    """Executa a Etapa 2: Classificação Multiclasse para Problema Multidimensional (RecFac)."""
    print("\n" + "="*70)
    print("ETAPA 2: Classificação Multiclasse (RecFac)")
    print("="*70)

    try:
        chosen_size = (40, 40)
        print(f"Dimensão da imagem escolhida: {chosen_size[0]}x{chosen_size[1]}")
        X, y = load_recfac(folder='recfac', choose_size=chosen_size)
        
        C = len(np.unique(y))
        print(f"Dados carregados: N={X.shape[0]}, p={X.shape[1]}, Classes={C}")
        
        rng_seed = 42
        R_val = 10
        
        hp_mlp = {'hidden_sizes': [50], 'activation': 'tanh', 'lr': 0.01, 'epochs': 100, 'random_state': rng_seed}
        hp_rbf = {'n_centers': 50, 'random_state': rng_seed}
        
        print(f"\n5. Executando Validação Monte Carlo (R={R_val}) com 80/20 para Multiclasse...")
        
        all_results = {}
        model_configs = {
            'MLP Multicamadas': {'name': 'mlp', **hp_mlp},
            'Rede RBF': {'name': 'rbf', **hp_rbf}
        }
        
        for name, config in model_configs.items():
            print(f"  -> Rodando {name} (Multiclasse)...")
            
            model_name = config['name']
            model_kwargs = {k: v for k, v in config.items() if k != 'name'}

            results = run_binary_classification_mc(X, y, 
                                                   model_name=model_name, 
                                                   R=R_val, 
                                                   test_size=0.2, 
                                                   rng_seed=rng_seed, 
                                                   **model_kwargs)
            all_results[name] = results
            print(f"  -> {name} concluído.")

        print("\n6. Plotando Matrizes de Confusão e Curvas de Aprendizado (Melhor Acurácia)...")
        for name, results in all_results.items():
            accuracies = [res['metrics']['accuracy'] for res in results]
            if not accuracies: continue

            max_acc_idx = np.argmax(accuracies)

            plot_confusion_matrix(results[max_acc_idx]['confusion'], results[max_acc_idx]['labels'], 
                                  title=f'{name} - Matriz de Confusão (Melhor Acc: {accuracies[max_acc_idx]:.4f})')
            plt.show()
            plot_learning_curve(results[max_acc_idx]['history'], title=f'{name} - Curva de Aprendizado (Melhor Acc)')
            plt.show()

        print("\n7. Calculando estatísticas resumo e gerando tabela/gráficos (R=10)...")
        metric = 'accuracy'
        metric_data = {}; summary_table = {}
        for name, results in all_results.items():
            values = [res['metrics'].get(metric, 0.0) for res in results if 'metrics' in res]
            if not values: continue
            metric_data[name] = values
            summary_table[name] = summary_statistics(values)

        format_summary_table(summary_table, metric)
        plot_box_violin_numpy(metric_data, metric, 'Etapa 2')

    except FileNotFoundError as e:
        print(f"ERRO: {e}. Por favor, verifique se a pasta 'recfac' e suas subpastas estão corretas.")
    except Exception as e:
        print(f"ERRO durante a Etapa 2: {e}")

def main_demo():
    print("Iniciando Modelo...")
    run_etapa1_classification_spiral()
    run_etapa2_multiclass_recfac()
    print("\nExecução concluída!")


if __name__ == '__main__':
    main_demo()