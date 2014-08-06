def erode(inputList):
    outputList = []
    for i, eachListItem in enumerate(inputList):
        try:
            prevItem = inputList[i-1]
            nextItem = inputList[i+1]
        except Exception as e:
            outputList.append(eachListItem)
        if eachListItem == prevItem or eachListItem == nextItem:
            outputList.append(eachListItem)
            continue
        else:
            outputList.append(prevItem)
    return outputList
