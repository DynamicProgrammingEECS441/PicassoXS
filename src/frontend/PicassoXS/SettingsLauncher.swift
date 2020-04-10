//
//  SettingsLauncher.swift
//  PicassoXFinal
//
//  Created by Senhao Wang on 4/9/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit



class SettingsLauncher: NSObject, UICollectionViewDataSource, UICollectionViewDelegate,UICollectionViewDelegateFlowLayout{
    
    let blackView = UIView()
    let settings : [String] = {
        return ["Portrait Mode","Regular","Cancel"]
    }()
    
    let collectionView: UICollectionView = {
        let layout = UICollectionViewFlowLayout()
        let cv = UICollectionView(frame: .zero, collectionViewLayout: layout)
        cv.backgroundColor = UIColor.white
        return cv
    }()
    
    let cellId = "cellId"
    
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return 3
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: cellId, for: indexPath) as! SettingCell
        let mode = settings[indexPath.item]
        cell.name = mode
        return cell
    }
    
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        return CGSize(width: collectionView.frame.width, height: 50)
    }
    
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        
        GlobalVariables.Mode = indexPath.item
    }
    

    
    
    
    
    func showSettings(){
           
              //Call animation
              if let window = UIApplication.shared.keyWindow{
                  blackView.backgroundColor = UIColor(white:0, alpha: 0.5)
                  blackView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(handleDismiss)))
                  
                  
                window.addSubview(blackView)
                window.addSubview(collectionView)
                
                let height: CGFloat = 200
                let y = window.frame.size.height - height
                collectionView.frame = CGRect(x: 0,y: window.frame.size.height,width: window.frame.size.width,height: height)
                
                blackView.frame = window.frame
                blackView.alpha = 0
                
                
                UIView.animate(withDuration: 0.5, delay: 0, usingSpringWithDamping: 1, initialSpringVelocity: 1, options: .curveEaseOut, animations: {
                    self.blackView.alpha = 1
                    self.collectionView.frame = CGRect(x: 0,y: y,width: window.frame.size.width,height: height)
                }, completion: nil)
                
            }
       }
       
       @objc func handleDismiss(){
            UIView.animate(withDuration: 0.5, animations: {
                self.blackView.alpha = 0
                
                if let window = UIApplication.shared.keyWindow{
                    self.collectionView.frame = CGRect(x: 0,y: window.frame.height, width: self.collectionView.frame.width, height: self.collectionView.frame.height)
                }
              })
       }
    
    
    override init(){
        super.init()
        
        collectionView.dataSource = self
        collectionView.delegate = self
        
        collectionView.register(SettingCell.self, forCellWithReuseIdentifier: cellId)
        
    }
    
   
   
}

