//
//  FilterViewCellCollectionViewCell.swift
//  PicassoXFinal
//
//  Created by Senhao Wang on 4/9/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import UIKit

class FilterViewCell: UICollectionViewCell {
    @IBOutlet weak var Image: UIImageView!
    
    @IBOutlet weak var name: UILabel!
    
    func setView(filter: Filter ){
        Image.image = filter.FilterImage
        name.text = filter.name
    }
    
       
}
