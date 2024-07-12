# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core.utils import ALGORITHMS
name2alg = ALGORITHMS

def get_algorithm(args, net_builder, tb_log, logger, teacher_net_builder=None, **kwargs):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm]( # name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger,
            teacher_net_builder=teacher_net_builder,
            **kwargs
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.algorithm)}')



