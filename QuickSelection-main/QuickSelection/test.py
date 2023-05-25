print("WHILE LOOP STARTED FOR TRAINING DATA")
class_indices_y_test = {class_label: np.where(adjusted_y_test == class_label)[0] for class_label in
                         np.unique(adjusted_y_test)}
target_counts_y_test = np.round(distribution * np.sum(adjusted_counts_y_test)).astype(int)
# Limit the target counts to not exceed the original counts
target_counts_y_test = np.minimum(target_counts_y_test, adjusted_counts_y_test)
print("target_counts _y_test loop", target_counts_y_test)

for class_label in np.unique(adjusted_y_test):
    print("class_label loop", class_label)
    class_indices_for_label_y_test = class_indices_y_test[class_label]
    print("class_indices_for_label y_test loop", class_indices_for_label_y_test)
    class_indices_sampled_y_test = np.random.choice(class_indices_for_label_y_test,
                                                     size=target_counts_y_test[class_label], replace=False)
    print("class_indices_sampled y_test loop", class_indices_sampled_y_test)
    adjusted_X_test.extend(adjusted_X_test[class_indices_sampled_y_test])
    adjusted_y_test.extend(adjusted_y_test[class_indices_sampled_y_test])

    # Update the adjusted counts based on the adjusted dataset
    print("len(class_indices_y_test[class_label] loop", len(class_indices_y_test[class_label]))
    adjusted_counts_y_test = np.array(
        [len([label for label in adjusted_y_test if label == class_label]) for class_label in
         np.unique(adjusted_y_test)])
    print("adjusted_counts_y_test loop", adjusted_counts_y_test)

    adjusted_X_test = np.array(adjusted_X_test)
    adjusted_y_test = np.array(adjusted_y_test)
    print("adjusted_X_test", adjusted_X_test)
    print("adjusted_y_test", adjusted_y_test)
    x = (adjusted_counts_y_test / np.sum(adjusted_counts_y_test))
    print("adjusted_counts_y_test / np.sum(adjusted_counts_y_test)", x)
    print("distribution", distribution)