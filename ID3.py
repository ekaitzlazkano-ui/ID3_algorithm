import math

class Node:
    def __init__(self, value):
        self.value = value
        self.children = {}

def ID3(examples: dict[tuple, str], target_attribute: str, attributes: list[str]) -> Node:
    #example of each parameter:
        #examples --> dataset to train the model --> format: { (Sunny, Windy): Negative , (Cloudy, Not wind): Positive } (target attribute does not have to be binary)
        #Target_Attribute --> Will we play tennis?
        #Attributes --> Sunny, Wind, Rain...
    
    root: Node = Node() #create a root node for the tree
    if all(value == next(iter(examples.values())) for value in examples.values()): #linea generada con ChatGPT // If all examples are positive, Return the single-node tree Root, with label = +. If all examples are negative, Return the single-node tree Root, with label = -.
        root.value = next(iter(examples.values()))
        return root
    if not attributes: #If number of predicting attributes is empty, then Return the single node tree Root with label = most common value of the target attribute in the examples
        count = {}
        for label in examples:
            if examples[label] in count:
                count[examples[label]]+=1
            else:
                count[examples[label]] = 1
        root.value = max(count, key=count.get) #linea generada con ChatGPT
        return root
    #Otherwise Begin
    max_attribute = ""
    max_info_gain = -1
    index = 0
    for i, attribute in enumerate(attributes):
        info_gain = _information_gain(examples, i)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_attribute = attribute #A ← The Attribute that best classifies examples.
            index = i
    root.value = max_attribute #Decision Tree attribute for Root = A
    values_attribute = set()
    for label in examples:
        values_attribute.add(label[index])
    for value in values_attribute: #For each possible value, vi, of A
        root.children[value] = 0 #Add a new tree branch below Root, corresponding to the test A = vi
        examples_vi = {} #Let Examples(vi) be the subset of examples that have the value vi for A
        for label in examples:
            if label[index]==value:
                new_label = label[:index] + label[index+1:]
                examples_vi[new_label] = examples[label]
        if not examples_vi: #If Examples(vi) is empty
            count = {}
            for label in examples:
                if examples[label] in count:
                    count[examples[label]]+=1
                else:
                    count[examples[label]] = 1
            root.children[value] = max(count, key=count.get) #Then below this new branch add a leaf node with label = most common target value in the examples
        else: #Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
            root.children[value] = ID3(examples_vi, target_attribute, attributes[:index]+attributes[index+1:]) 
    #End
    return root #Return Root
    
def _information_gain(examples: dict[tuple, str], attribute_index: int):
    n = len(examples)
    total_entropy = _entropy(examples)
    values = {} #"values" contains a dictionary of the possible values the attribute can take and the amount of each
    for label in examples:
        if(label[attribute_index] in values):
            values[label[attribute_index]]+=1
        else:
            values[label[attribute_index]]=1
    result = total_entropy
    for value in values:
        subset = {}
        for label in examples:
            if label[attribute_index]==value:
                subset[label] = examples[label]
        result -= (values[value]/n)*_entropy(subset)

    return result

def _entropy(examples: dict[tuple, str]):
    n = len(examples)
    values = {}
    for label in examples:
        if examples[label] in values:
            values[examples[label]]+=1
        else:
            values[examples[label]] = 1
    result = 0
    for value in values.values():
        p = value/n
        result += -p*math.log2(p)
    return result
