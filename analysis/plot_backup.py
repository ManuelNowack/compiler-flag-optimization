for module in runtimes.columns:
    ax = runtimes[[module]].transpose().plot(kind="bar")
    plt.savefig(f"plots/runtime/{module}", bbox_inches="tight")
    plt.close(ax.figure)
    # plt.bar(
    #     range(len(runtimes)),
    #     runtimes[module],
    #     color=plt.cm.Paired(np.arange(len(df))),
    #     tick_label=runtimes.index)
    # plt.savefig(f"plots/runtime/{module}")
    # plt.close()
# chunks = collections.defaultdict(list)
# for module in runtimes.columns:
#     program, dataset, command = module.split(":")
#     chunks[f"{program}:{command}"].append(module)
# for name, modules in chunks.items():
#     ax = runtimes[modules].transpose().rename(lambda x: x.split(":")[1]).plot(kind="bar")
#     plt.savefig(f"plots/runtime/{name}", bbox_inches="tight")
#     plt.close(ax.figure)