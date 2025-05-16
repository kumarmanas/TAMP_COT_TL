import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog='cottl',
        description='Converts planning instructions to LTL'
    )
    parser.add_argument('--keydir', required=False, default="", help='Folder where Key is stored')
    parser.add_argument('--prompt', required=False, default="totsrl", help='Your prompt .txt file')
    parser.add_argument('--maxtokens', required=False, default=500, help='Maximum number of tokens to compute')
    parser.add_argument('--model', required=False, default="deepseek", help='Specify the model (default: "deepseek")')
    parser.add_argument('--plan_ins', required=True, default="", help='Input planning text') 
    parser.add_argument('--fewshots', required=False, default="",  help='Prompts')
    parser.add_argument('--keyfile', required=False, default="", help='Hf and open AI GPT key')
    parser.add_argument('--given_translations', required=False, default="", help='Provides given translations')
    parser.add_argument('--num_tries', type=int, required=False, default=3, help="Number of runs the underlying language model attempts a translation.")
    parser.add_argument('--temperature', required=False, default=0.7, type=float, help="Model temperature range 0 to 2")
    parser.add_argument('--model_size', type=str, choices=['small', 'large'], default='small',
                       help='Size of model to use (small=1.5B, large=32B)')
    
    return parser.parse_args()