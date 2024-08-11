# def test_trainer_fast_dev_run():
#     trainer = L.Trainer(fast_dev_run=True)
#     trainer.fit(lit_conv, train_dl, valid_dl)


# def test_trainer_overfit_batches():
#     trainer = L.Trainer(overfit_batches=1, max_epochs=50)
#     trainer.fit(lit_conv, train_dl)
#     final_accuracy = trainer.callback_metrics["train_acc_epoch"].item()
#     assert final_accuracy > 0.99
