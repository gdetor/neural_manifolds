from common import bringDirectoryToLife

# Example usage
main_results_dir = "./results/"
print(f"Checking if dir: {main_results_dir} exists, if not I'll create it")
bringDirectoryToLife(main_results_dir)

tda_results_dir = "./tda_results/"
print(f"Checking if dir: {tda_results_dir} exists, if not I'll create it")
bringDirectoryToLife(tda_results_dir)

convex_hull_results_dir = "./convex_hull_results/"
print(f"Checking if dir: {convex_hull_results_dir} exists, if not I'll create it")
bringDirectoryToLife(convex_hull_results_dir)
