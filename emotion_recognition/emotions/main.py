from lightning.pytorch.cli import LightningCLI
from clearml import Task
from emotion_recognition.emotions.lightning_module import MyModel
from emotion_recognition.emotions.lightning_data_module import MyDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Program-level
        parser.add_argument("--project_name", type=str, default="EMO")
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="Train",
        )
        parser.add_argument("--log_to_clearml", type=bool, default=True)
        parser.add_argument("--clearml_tags", type=list, default=[])
        parser.add_argument("--task_id", type=str, default="")
        parser.add_argument("--ckpt_path", type=str, default="")
        parser.add_argument(
            "--clearml_api_host",
            type=str,
            default="x",
        )
        parser.add_argument(
            "--clearml_web_host",
            type=str,
            default="x",
        )
        parser.add_argument(
            "--clearml_files_host",
            type=str,
            default="x",
        )

    @staticmethod
    def activate_clearml(config, task_type="training"):
        Task.set_credentials(
            api_host=config["clearml_api_host"],
            web_host=config["clearml_web_host"],
            files_host=config["clearml_files_host"],
        )

        if config["task_id"]:
            task = Task.init(
                reuse_last_task_id=config["task_id"],
                continue_last_task=0,
                task_type=task_type,
            )
            print(f"Reuse task: {task.id}")
        else:
            task = Task.init(
                project_name=config["project_name"],
                task_name=config["experiment_name"],
                task_type=task_type,
            )
            print(f"Create task: {task.id}")
        task.add_tags(config["clearml_tags"])

        return task


def main():
    cli = MyLightningCLI(
        MyModel, MyDataModule, save_config_kwargs={"overwrite": True}, run=False
    )

    if cli.config["log_to_clearml"] and cli.trainer.local_rank == 0:
        clearml_task = cli.activate_clearml(cli.config, task_type="training")
        cli.config["task_id"] = clearml_task.id
        config_dict = dict(cli.config)
        clearml_task.set_parameters(config_dict)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
        ckpt_path=cli.config["ckpt_path"],
    )
    cli.trainer.test(datamodule=cli.datamodule, ckpt_path=cli.config["ckpt_path"])


if __name__ == "__main__":
    main()
