import torch
from collections import defaultdict



class BiasLogger(object):
    def __init__(self,envs,info=None):
        self.info = info
        self.label_results = []
        self.degree_results = []
        self.envs_results = [[] for _ in range(envs)]
        self.envs = envs
    def add_result(self,label_results,degree_results):
        self.label_results.append(label_results)
        self.degree_results.append(degree_results)
    def add_envs_result(self,envs_result):
        for i in range(self.envs):
            self.envs_results[i].append(envs_result[i])
    def print_statistics(self,mylog=None,run=None):

        if run is None:
            assert len(self.label_results) > 0
            label_accuracys = [[] for _ in range(len(self.label_results[0]))]
            for i in range(len(label_accuracys)):
                for j in range(len(self.label_results)):
                    if i == len(label_accuracys) - 1:
                        label_accuracys[i].append(self.label_results[j].get('avg'))
                    else:
                        if i in self.label_results[j].keys():
                            label_accuracys[i].append(self.label_results[j].get(i))

            print()
            print(file=mylog)
            print(f'label accuracy:')
            print(f'label accuracy:',file=mylog)
            for label in range(len(label_accuracys)):
                if label == len(label_accuracys) - 1:
                    print(
                        f'total label accuracy: {(100 * torch.tensor(label_accuracys[label])).mean():.2f} ± {(100 * torch.tensor(label_accuracys[label])).std():.2f}')
                    print(
                        f'total label accuracy: {(100 * torch.tensor(label_accuracys[label])).mean():.2f} ± {(100 * torch.tensor(label_accuracys[label])).std():.2f}',
                        file=mylog)
                else:
                    print(
                        f'label {label} accuracy: {(100 * torch.tensor(label_accuracys[label])).mean():.2f} ± {(100 * torch.tensor(label_accuracys[label])).std():.2f}')
                    print(
                        f'label {label} accuracy: {(100 * torch.tensor(label_accuracys[label])).mean():.2f} ± {(100 * torch.tensor(label_accuracys[label])).std():.2f}',
                        file=mylog)

            assert len(self.degree_results) > 0
            degree_accuracys = [[] for _ in range(len(self.degree_results[0]))]
            for i in range(len(degree_accuracys)):
                for j in range(len(self.degree_results)):
                    degrees = list(self.degree_results[j].keys())
                    degree_accuracys[i].append(self.degree_results[j].get(degrees[i]))

            degree_accuracys = 100 * torch.tensor(degree_accuracys)
            degrees = list(self.degree_results[j].keys())
            print()
            print(file=mylog)
            print(f'degree accuracy:')
            print(f'degree accuracy:',file=mylog)
            for degree in range(len(degree_accuracys)):
                print(f'degree{degrees[degree]:.2f} accuracy: {degree_accuracys[degree].mean():.2f} ± {degree_accuracys[degree].std():.2f}')
                print(
                    f'degree{degrees[degree]:.2f} accuracy: {degree_accuracys[degree].mean():.2f} ± {degree_accuracys[degree].std():.2f}',file=mylog)


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self,mylog=None, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')

            print(f'Run {run + 1:02d}:',file=mylog)
            print(f'Highest Train: {result[:, 0].max():.2f}',file=mylog)
            print(f'Highest Valid: {result[:, 1].max():.2f}',file=mylog)
            print(f'  Final Train: {result[argmax, 0]:.2f}',file=mylog)
            print(f'   Final Test: {result[argmax, 2]:.2f}',file=mylog)
        else:

            best_results = []
            for r in self.results:
                r = torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))
            best_result = 100 * torch.tensor(best_results)

            print(f'All runs:')
            print(f'All runs:',file=mylog)
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}',file=mylog)
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}',file=mylog)
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}',file=mylog)
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}',file=mylog)
            return best_result[:, 1], best_result[:, 3]


class SimpleLogger(object):
    def __init__(self, desc, param_names, num_values=2):
        self.results = defaultdict(dict)
        self.param_names = tuple(param_names)
        self.used_args = list()
        self.desc = desc
        self.num_values = num_values
    
    def add_result(self, run, args, values):
        assert(len(args) == len(self.param_names))
        assert(len(values) == self.num_values)
        self.results[run][args] = values
        if args not in self.used_args:
            self.used_args.append(args)
    
    def get_best(self, top_k=1):
        all_results = []
        for args in self.used_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)[-1]
            results_std = results.std(dim=0)

            all_results.append((args, results_mean))
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [i[0] for i in results]
            
    def prettyprint(self, x):
        if isinstance(x, float):
            return '%.2f' % x
        return str(x)
        
    def display(self, args = None):
        
        disp_args = self.used_args if args is None else args
        if len(disp_args) > 1:
            print(f'{self.desc} {self.param_names}, {len(self.results.keys())} runs')
        for args in disp_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)
            results_std = results.std(dim=0)
            res_str = f'{results_mean[0]:.2f} ± {results_std[0]:.2f}'
            for i in range(1, self.num_values):
                res_str += f' -> {results_mean[i]:.2f} ± {results_std[1]:.2f}'
            print(f'Args {[self.prettyprint(x) for x in args]}: {res_str}')
        if len(disp_args) > 1:
            print()
        return results
