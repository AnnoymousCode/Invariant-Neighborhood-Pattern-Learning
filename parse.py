from models import LINKX,INPL

def parse_method(args, dataset, n, c, d, device):
    if args.method == 'inpl':
        model = INPL(   num_nodes=n,
                             nfeat=d,
                             nhidden=args.hidden_channels,
                             nclass=c,
                             dropout=args.dropout,
                             lamda=args.lamda,
                             alpha=args.alpha,
                             num_layers=args.num_layers,
                             neighbors=args.neighbors,
                             nlayers=args.layer, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
    elif args.method == 'linkx':
        model = LINKX(d, args.hidden_channels, c, args.num_layers, dataset.graph['num_nodes'],
        inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, dropout=args.dropout, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='squirrel')#fb100/Penn94、twitch-gamer、film、squirrel、chameleon、wisconsin、cornell、texas
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--method', '-m', type=str, default='inpl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')
    parser.add_argument("--SGD", action='store_true', help='Use SGD as optimizer')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)
    parser.add_argument('--cluster',type=str, default='knn',help='cluster method')
    parser.add_argument('--gpu',type=int, default=0,help='train on which gpu')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.1, help='lamda.')
    parser.add_argument('--layer', type=int, default=1, help='Number of layers.')
    parser.add_argument('--result_path', type=str, default='result.txt', help='output text')
    parser.add_argument('--gumblesoftmax',type=int,default=1,help='gumblesoftmax or softmax')
    parser.add_argument('--tau',type=int,default=2,help='tau for gmblesoftmax')
    parser.add_argument('--lianxu',type=int,default=0,help='0/1 or add = 1')
    parser.add_argument('--ret_l',type=float,default=0.5)
    parser.add_argument('--envs',type=int,default=1,help = 'the quantities of envorinments')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='weight for mean of risks from multiple domains')
    parser.add_argument('--hx', type=int, default=1,help='cluster by hx or hA')
    parser.add_argument('--xunhuan', type=int, default=1 ,help='Number of times to divide the environment')
    parser.add_argument('--epochs1', type=int, default=2,help='epochs for gnn train by x')
    parser.add_argument('--epochs2', type=int, default=2,help='epochs for gnn train by hx/ha')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--neighbors', type=int, default=1)