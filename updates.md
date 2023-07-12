# Update

## Version 2.1.2
- Check GRIDAY is vailed or not when you run `install_griday`
- `torchmetrics < 1.0.0` in requirements
- Fix bugs in predict when loss is `classification` and n_classes are 2.

## Version 2.1.1
- Fixed a bug when the structure name of raw_[downstream].json contains a cif during prepare_data.
- Changed an error that occurred when there were multiple devices in an interactive environment to a warning, automatically converting the configuration to a single device.
- Changed a bug for command-line interface (moftransformer run)
- Change float type 16 -> 32 before denormalization.
- Add command-line interface (moftransformer predict)
- Modified minor typo errors in document
- Remove logger file in function "predict"
- Make "test" function
- Fix 'predict' function in classification task -> add "classification_logits_index'
- 'best.ckpt' are generated after training
- Update Readme.md to include the comment regarding only support for Linux