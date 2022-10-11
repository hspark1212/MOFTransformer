import argparse
import textwrap
from importlib import import_module
from argparse import RawTextHelpFormatter
from moftransformer import __version__

commands = [
    #('info' , 'moftransformer.cli.info'),
    ('install-griday' , 'moftransformer.cli.install_griday'),
    ('uninstall-griday', 'moftransformer.cli.uninstall_griday'),
    ('run', 'moftransformer.cli.run'),
    ('download', 'moftransformer.cli.download'),
]


def main(prog='moftransformer', version=__version__, commands=commands, args=None):
    parser = argparse.ArgumentParser(prog=prog, )
    parser.add_argument('--version', action='version',
                        version= '%(prog)s-{}'.format(version))
    parser.add_argument('-T', '--traceback', action='store_true')
    subparsers = parser.add_subparsers(title='Sub-command', dest='command')

    subparser = subparsers.add_parser('help',
                                      description='Help',
                                      help='Help for sub-command.')

    subparser.add_argument('helpcommand',
                           nargs='?',
                           metavar='sub-command',
                           help='Provide help for sub-command.')


    functions = {}
    parsers = {}
    for command, module_name in commands:
        cmd = import_module(module_name).CLICommand
        docstring = cmd.__doc__
        if docstring is None:
            # Backwards compatibility with GPAW
            short = cmd.short_description
            long = getattr(cmd, 'description', short)
        else:
            parts = docstring.split('\n', 1)
            if len(parts) == 1:
                short = docstring
                long = docstring
            else:
                short, body = parts
                long = short
                #long = short + '\n' + textwrap.dedent(body)
        subparser = subparsers.add_parser(
            command,
            formatter_class=RawTextHelpFormatter,
            help=short,
            description=long)
        cmd.add_arguments(subparser)
        functions[command] = cmd.run
        parsers[command] = subparser

    #if hook:
    #    args = hook(parser, args)
    #    args = hook(parser, args)
    #else:
    args = parser.parse_args(args)

    if args.command == 'help':
        if args.helpcommand is None:
            parser.print_help()
        else:
            parsers[args.helpcommand].print_help()
    elif args.command is None:
        parser.print_usage()
    else:
        f = functions[args.command]
        try:
            if f.__code__.co_argcount == 1:
                f(args)
            else:
                f(args, parsers[args.command])
        except KeyboardInterrupt:
            pass
        except Exception as x:
            if args.traceback:
                raise
            else:
                l1 = '{}: {}\n'.format(x.__class__.__name__, x)
                l2 = ('To get a full traceback, use: {} -T {} ...'
                      .format(prog, args.command))
                parser.error(l1 + l2)