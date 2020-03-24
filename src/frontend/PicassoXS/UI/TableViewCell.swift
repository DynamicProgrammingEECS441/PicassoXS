//
//  TableViewCell.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/22/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import UIKit



class TableViewCell: UITableViewCell {

    @IBOutlet weak var ImageCell: UIImageView!
    @IBOutlet weak var NameLabel: UILabel!

    func setFilter(filter: Filter){
        ImageCell.image = filter.FilterImage
        NameLabel.text = filter.name
    }
    
}
